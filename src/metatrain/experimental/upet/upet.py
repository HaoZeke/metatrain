"""UPET: Clean PyTorch reimplementation of metatrain PET.

Uses rectangular neighbor list format compatible with JAX PET for benchmarking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cutoff_bump(r, cutoff, width=0.5):
    x = (r - (cutoff - width)) / width
    x_safe = torch.clamp(x, 1e-6, 1 - 1e-6)
    bump = 0.5 * (1 + torch.tanh(1 / torch.tan(torch.pi * x_safe)))
    return torch.where(x <= 0, 1.0, torch.where(x >= 1, 0.0, bump))


def rms_norm(x, weight, eps=1e-6):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


class Attention(nn.Module):
    def __init__(self, dim, num_heads, temperature=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / (self.head_dim**0.5 * temperature)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, cutoffs):
        B, T, _ = x.shape
        H, d = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        bias = torch.log(cutoffs.clamp(min=1e-15))[:, None, :, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)

        return self.out(out.transpose(1, 2).reshape(B, T, -1))


class TransformerLayer(nn.Module):
    def __init__(self, d_pet, d_node, d_ff, num_heads, norm, activation, trans_type, temp):
        super().__init__()
        self.trans_type = trans_type
        self.attention = Attention(d_pet, num_heads, temp)

        self.norm_attn = nn.Parameter(torch.ones(d_pet)) if norm == "RMSNorm" else None
        self.norm_attn_ln = nn.LayerNorm(d_pet) if norm != "RMSNorm" else None
        self.norm_mlp = nn.Parameter(torch.ones(d_pet)) if norm == "RMSNorm" else None
        self.norm_mlp_ln = nn.LayerNorm(d_pet) if norm != "RMSNorm" else None

        self.mlp_in = nn.Linear(d_pet, 2 * d_ff if activation == "SwiGLU" else d_ff)
        self.mlp_out = nn.Linear(d_ff, d_pet)
        self.is_swiglu = activation == "SwiGLU"

        self.expanded = d_node != d_pet
        if self.expanded:
            self.center_contract = nn.Linear(d_node, d_pet)
            self.center_expand = nn.Linear(d_pet, d_node)
            self.norm_center = (
                nn.Parameter(torch.ones(d_node)) if norm == "RMSNorm" else None
            )
            self.norm_center_ln = nn.LayerNorm(d_node) if norm != "RMSNorm" else None
            self.center_mlp_in = nn.Linear(
                d_node, 2 * 2 * d_node if activation == "SwiGLU" else 2 * d_node
            )
            self.center_mlp_out = nn.Linear(2 * d_node, d_node)

    def _norm(self, x, weight, ln):
        return rms_norm(x, weight) if weight is not None else ln(x)

    def _mlp(self, x):
        if self.is_swiglu:
            v, g = self.mlp_in(x).chunk(2, dim=-1)
            return self.mlp_out(v * torch.sigmoid(g))
        return self.mlp_out(F.silu(self.mlp_in(x)))

    def _center_mlp(self, x):
        if self.is_swiglu:
            v, g = self.center_mlp_in(x).chunk(2, dim=-1)
            return self.center_mlp_out(v * torch.sigmoid(g))
        return self.center_mlp_out(F.silu(self.center_mlp_in(x)))

    def forward(self, node, edge, cutoffs):
        proj = self.center_contract(node) if self.expanded else node
        tokens = torch.cat([proj, edge], dim=1)

        if self.trans_type == "PreLN":
            normed = self._norm(tokens, self.norm_attn, self.norm_attn_ln)
            attn = self.attention(normed, cutoffs)
            out_node, out_edge = attn[:, :1], attn[:, 1:]

            if self.expanded:
                out_node = node + self.center_expand(out_node)
                normed_node = self._norm(out_node, self.norm_center, self.norm_center_ln)
                out_node = out_node + self._center_mlp(normed_node)
            else:
                out_node = node + out_node

            out_edge = edge + out_edge
            out_edge = out_edge + self._mlp(
                self._norm(out_edge, self.norm_mlp, self.norm_mlp_ln)
            )
        else:  # PostLN
            attn = self.attention(tokens, cutoffs)
            tokens = self._norm(tokens + attn, self.norm_attn, self.norm_attn_ln)
            tokens = self._norm(tokens + self._mlp(tokens), self.norm_mlp, self.norm_mlp_ln)
            out_node, out_edge = tokens[:, :1], tokens[:, 1:]
            if self.expanded:
                out_node = node + self.center_expand(out_node)
                normed_node = self._norm(out_node, self.norm_center, self.norm_center_ln)
                out_node = out_node + self._center_mlp(normed_node)

        return out_node, out_edge


class UPET(nn.Module):
    def __init__(
        self,
        d_pet=128,
        d_node=512,
        d_head=128,
        d_feedforward=256,
        num_heads=8,
        num_attention_layers=1,
        num_gnn_layers=2,
        cutoff=7.5,
        cutoff_width=0.5,
        cutoff_function="Bump",
        normalization="RMSNorm",
        activation="SwiGLU",
        transformer_type="PreLN",
        attention_temperature=1.0,
        featurizer_type="feedforward",
        num_species=103,
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.cutoff_width = cutoff_width
        self.d_pet = d_pet
        self.d_node = d_node
        self.featurizer_type = featurizer_type
        self.num_gnn_layers = num_gnn_layers

        self.register_buffer("probes", torch.arange(0.5, cutoff, cutoff_width / 4))

        self.edge_embedder = nn.Embedding(num_species, d_pet)

        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            layer = nn.Module()
            layer.edge_embed = nn.Linear(4, d_pet)
            layer.neighbor_embed = nn.Embedding(num_species, d_pet) if i > 0 else None
            n_in = 2 * d_pet if i == 0 else 3 * d_pet
            layer.compress = nn.Sequential(
                nn.Linear(n_in, d_pet), nn.SiLU(), nn.Linear(d_pet, d_pet)
            )
            layer.trans = nn.Module()
            layer.trans.layers = nn.ModuleList(
                [
                    TransformerLayer(
                        d_pet,
                        d_node,
                        d_feedforward,
                        num_heads,
                        normalization,
                        activation,
                        transformer_type,
                        attention_temperature,
                    )
                    for _ in range(num_attention_layers)
                ]
            )
            self.gnn_layers.append(layer)

        if featurizer_type == "feedforward":
            self.node_embedders = nn.ModuleList([nn.Embedding(num_species, d_node)])
            self.comb_norms = nn.ModuleList(
                [nn.LayerNorm(2 * d_pet) for _ in range(num_gnn_layers)]
            )
            self.comb_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * d_pet, 2 * d_pet),
                        nn.SiLU(),
                        nn.Linear(2 * d_pet, d_pet),
                    )
                    for _ in range(num_gnn_layers)
                ]
            )
            self.num_readouts = 1
        else:
            self.node_embedders = nn.ModuleList(
                [nn.Embedding(num_species, d_node) for _ in range(num_gnn_layers)]
            )
            self.comb_norms = nn.ModuleList()
            self.comb_mlps = nn.ModuleList()
            self.num_readouts = num_gnn_layers

        self.node_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_node, d_head),
                    nn.SiLU(),
                    nn.Linear(d_head, d_head),
                    nn.SiLU(),
                )
                for _ in range(self.num_readouts)
            ]
        )
        self.edge_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_pet, d_head),
                    nn.SiLU(),
                    nn.Linear(d_head, d_head),
                    nn.SiLU(),
                )
                for _ in range(self.num_readouts)
            ]
        )
        self.node_last = nn.ModuleList(
            [nn.Linear(d_head, 1) for _ in range(self.num_readouts)]
        )
        self.edge_last = nn.ModuleList(
            [nn.Linear(d_head, 1) for _ in range(self.num_readouts)]
        )

    def forward(
        self,
        R_ij,
        centers,
        neighbors,
        species,
        reverse,
        pair_mask,
        atom_mask,
        pair_cutoffs=None,
    ):
        P = R_ij.shape[0]
        N = species.shape[0]
        n = P // N

        # Ensure masks are float with same dtype as R_ij
        pair_mask_f = pair_mask.to(R_ij.dtype)
        atom_mask_f = atom_mask.to(R_ij.dtype)

        r_ij = torch.sqrt((R_ij**2).sum(-1) + 1e-15)

        cutoff_val = pair_cutoffs if pair_cutoffs is not None else self.cutoff
        cutoffs = cutoff_bump(r_ij, cutoff_val, self.cutoff_width) * pair_mask_f

        R_ij_2d = R_ij.reshape(N, n, 3)
        r_ij_2d = r_ij.reshape(N, n)
        cutoffs_2d = cutoffs.reshape(N, n)
        pair_mask_2d = pair_mask_f.reshape(N, n)
        neighbor_species = species[neighbors].reshape(N, n)

        central = torch.ones(N, 1, device=R_ij.device, dtype=R_ij.dtype)
        cutoffs_ext = torch.cat([central, cutoffs_2d], dim=1)
        mask_ext = torch.cat([atom_mask_f[:, None], pair_mask_2d], dim=1).bool()
        cutoffs_ext = torch.where(mask_ext, cutoffs_ext, torch.zeros_like(cutoffs_ext))
        cutoff_matrix = cutoffs_ext[:, None, :].expand(N, 1 + n, -1)

        edge_msg = self.edge_embedder(neighbor_species) * pair_mask_2d[..., None]

        if self.featurizer_type == "feedforward":
            node_emb = self.node_embedders[0](species) * atom_mask_f[..., None]

            for idx, (gnn, norm, mlp) in enumerate(
                zip(self.gnn_layers, self.comb_norms, self.comb_mlps)
            ):
                edge_geom = torch.cat([R_ij_2d, r_ij_2d[..., None]], dim=-1)
                edge_feat = gnn.edge_embed(edge_geom) * pair_mask_2d[..., None]

                if idx == 0:
                    tokens = gnn.compress(torch.cat([edge_feat, edge_msg], dim=-1))
                else:
                    neighbor_feat = (
                        gnn.neighbor_embed(neighbor_species) * pair_mask_2d[..., None]
                    )
                    tokens = gnn.compress(
                        torch.cat([edge_feat, neighbor_feat, edge_msg], dim=-1)
                    )

                node_tok = node_emb[:, None, :]
                for trans_layer in gnn.trans.layers:
                    node_tok, tokens = trans_layer(node_tok, tokens, cutoff_matrix)
                node_emb = node_tok[:, 0, :]

                tokens_flat = tokens.reshape(P, -1)
                reversed_flat = tokens_flat[reverse]
                reversed_2d = reversed_flat.reshape(N, n, -1)
                edge_msg = (
                    edge_msg + tokens + mlp(norm(torch.cat([tokens, reversed_2d], dim=-1)))
                )

            node_feats, edge_feats = [node_emb], [edge_msg]

        else:  # residual
            node_feats, edge_feats = [], []

            for idx, (node_embed, gnn) in enumerate(
                zip(self.node_embedders, self.gnn_layers)
            ):
                node_emb = node_embed(species) * atom_mask_f[..., None]

                edge_geom = torch.cat([R_ij_2d, r_ij_2d[..., None]], dim=-1)
                edge_feat = gnn.edge_embed(edge_geom) * pair_mask_2d[..., None]

                if idx == 0:
                    tokens = gnn.compress(torch.cat([edge_feat, edge_msg], dim=-1))
                else:
                    neighbor_feat = (
                        gnn.neighbor_embed(neighbor_species) * pair_mask_2d[..., None]
                    )
                    tokens = gnn.compress(
                        torch.cat([edge_feat, neighbor_feat, edge_msg], dim=-1)
                    )

                node_tok = node_emb[:, None, :]
                for trans_layer in gnn.trans.layers:
                    node_tok, tokens = trans_layer(node_tok, tokens, cutoff_matrix)

                node_feats.append(node_tok[:, 0, :])
                edge_feats.append(tokens)

                tokens_flat = tokens.reshape(P, -1)
                reversed_flat = tokens_flat[reverse]
                reversed_2d = reversed_flat.reshape(N, n, -1)
                edge_msg = 0.5 * (edge_msg + reversed_2d)

        predictions = torch.zeros(N, device=species.device, dtype=R_ij.dtype)
        for i, (nf, ef) in enumerate(zip(node_feats, edge_feats)):
            node_pred = self.node_last[i](self.node_heads[i](nf))[:, 0]
            edge_pred = (
                self.edge_last[i](self.edge_heads[i](ef))[:, :, 0] * cutoffs_2d
            ).sum(1)
            predictions = predictions + node_pred + edge_pred

        return predictions * atom_mask_f


def predict(model, batch, compute_forces=True, compute_stress=False):
    """Compute energy, forces, stress."""
    need_grad = compute_forces or compute_stress
    R_ij = batch["R_ij"]
    if need_grad:
        R_ij = R_ij.clone().requires_grad_(True)

    species = batch["species"]
    centers, neighbors = batch["centers"], batch["neighbors"]
    reverse = batch["reverse"]
    pair_mask = batch["pair_mask"]
    atom_mask = batch["atom_mask"]
    atom_to_struct = batch["atom_to_structure"]
    N, num_struct = batch["num_atoms"], batch["num_structures"]
    
    device, dtype = R_ij.device, R_ij.dtype

    per_atom_energy = model(
        R_ij, centers, neighbors, species, reverse, pair_mask, atom_mask
    )

    energy = torch.zeros(num_struct + 1, device=device, dtype=dtype)
    energy.index_add_(0, atom_to_struct, per_atom_energy)
    results = {"energy": energy[:num_struct]}

    if need_grad:
        total_energy = per_atom_energy.sum()
        (dR,) = torch.autograd.grad(total_energy, R_ij, create_graph=model.training)
        
        if compute_forces:
            F1 = torch.zeros(N, 3, device=device, dtype=dtype)
            F2 = torch.zeros(N, 3, device=device, dtype=dtype)
            F1.index_add_(0, centers, dR)
            F2.index_add_(0, neighbors, dR)
            forces = F2 - F1
            # Return forces for real atoms only
            results["forces"] = forces[batch["atom_mask"]]
            
        if compute_stress:
            pair_struct = atom_to_struct[centers]
            stress = torch.zeros(num_struct + 1, 3, 3, device=device, dtype=dtype)
            stress.index_add_(
                0,
                pair_struct,
                torch.einsum("pa,pb->pab", R_ij, dR),
            )
            results["stress"] = stress[:num_struct]

    return results
