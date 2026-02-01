"""UPET: Clean PyTorch reimplementation of metatrain PET.

Uses rectangular neighbor list format compatible with JAX PET for benchmarking.
TorchScript compatible for metatomic export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


def cutoff_bump(r: torch.Tensor, cutoff: float, width: float = 0.5) -> torch.Tensor:
    x = (r - (cutoff - width)) / width
    x_safe = torch.clamp(x, 1e-6, 1 - 1e-6)
    bump = 0.5 * (1 + torch.tanh(1 / torch.tan(torch.pi * x_safe)))
    return torch.where(x <= 0, 1.0, torch.where(x >= 1, 0.0, bump))


class RMSNorm(nn.Module):
    """RMSNorm as a proper module for TorchScript compatibility."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, temperature: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / (self.head_dim**0.5 * temperature)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cutoffs: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.num_heads
        d = self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        bias = torch.log(cutoffs.clamp(min=1e-15))[:, None, :, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)

        return self.out(out.transpose(1, 2).reshape(B, T, -1))


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_pet: int,
        d_node: int,
        d_ff: int,
        num_heads: int,
        norm: str,
        activation: str,
        trans_type: str,
        temp: float,
    ):
        super().__init__()
        self.trans_type = trans_type
        self.attention = Attention(d_pet, num_heads, temp)
        self.use_rmsnorm = norm == "RMSNorm"
        self.is_swiglu = activation == "SwiGLU"
        self.expanded = d_node != d_pet

        if self.use_rmsnorm:
            self.norm_attn = RMSNorm(d_pet)
            self.norm_mlp = RMSNorm(d_pet)
        else:
            self.norm_attn = nn.LayerNorm(d_pet)
            self.norm_mlp = nn.LayerNorm(d_pet)

        self.mlp_in = nn.Linear(d_pet, 2 * d_ff if self.is_swiglu else d_ff)
        self.mlp_out = nn.Linear(d_ff, d_pet)

        if self.expanded:
            self.center_contract = nn.Linear(d_node, d_pet)
            self.center_expand = nn.Linear(d_pet, d_node)
            if self.use_rmsnorm:
                self.norm_center = RMSNorm(d_node)
            else:
                self.norm_center = nn.LayerNorm(d_node)
            self.center_mlp_in = nn.Linear(
                d_node, 2 * 2 * d_node if self.is_swiglu else 2 * d_node
            )
            self.center_mlp_out = nn.Linear(2 * d_node, d_node)
        else:
            self.center_contract = nn.Identity()
            self.center_expand = nn.Identity()
            self.norm_center = nn.Identity()
            self.center_mlp_in = nn.Identity()
            self.center_mlp_out = nn.Identity()

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_swiglu:
            h = self.mlp_in(x)
            h1, h2 = h.chunk(2, dim=-1)
            return self.mlp_out(h1 * torch.sigmoid(h2))
        return self.mlp_out(F.silu(self.mlp_in(x)))

    def _center_mlp(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_swiglu:
            h = self.center_mlp_in(x)
            h1, h2 = h.chunk(2, dim=-1)
            return self.center_mlp_out(h1 * torch.sigmoid(h2))
        return self.center_mlp_out(F.silu(self.center_mlp_in(x)))

    def forward(
        self, node: torch.Tensor, edge: torch.Tensor, cutoffs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.expanded:
            proj = self.center_contract(node)
        else:
            proj = node
        tokens = torch.cat([proj, edge], dim=1)

        if self.trans_type == "PreLN":
            normed = self.norm_attn(tokens)
            attn = self.attention(normed, cutoffs)
            out_node = attn[:, :1]
            out_edge = attn[:, 1:]

            if self.expanded:
                out_node = node + self.center_expand(out_node)
                normed_node = self.norm_center(out_node)
                out_node = out_node + self._center_mlp(normed_node)
            else:
                out_node = node + out_node

            out_edge = edge + out_edge
            out_edge = out_edge + self._mlp(self.norm_mlp(out_edge))
        else:  # PostLN
            attn = self.attention(tokens, cutoffs)
            tokens = self.norm_attn(tokens + attn)
            tokens = self.norm_mlp(tokens + self._mlp(tokens))
            out_node = tokens[:, :1]
            out_edge = tokens[:, 1:]
            if self.expanded:
                out_node = node + self.center_expand(out_node)
                normed_node = self.norm_center(out_node)
                out_node = out_node + self._center_mlp(normed_node)

        return out_node, out_edge


class GNNLayer(nn.Module):
    """Single GNN layer with its own combination MLP."""

    def __init__(
        self,
        d_pet: int,
        d_node: int,
        d_feedforward: int,
        num_heads: int,
        num_attention_layers: int,
        normalization: str,
        activation: str,
        transformer_type: str,
        attention_temperature: float,
        num_species: int,
        is_first: bool,
    ):
        super().__init__()
        self.is_first = is_first
        self.edge_embed = nn.Linear(4, d_pet)
        
        if not is_first:
            self.neighbor_embed = nn.Embedding(num_species, d_pet)
        else:
            self.neighbor_embed = nn.Embedding(1, d_pet)

        n_in = 2 * d_pet if is_first else 3 * d_pet
        self.compress = nn.Sequential(
            nn.Linear(n_in, d_pet), nn.SiLU(), nn.Linear(d_pet, d_pet)
        )

        self.trans_layers = nn.ModuleList(
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
        
        # Include combination norm and mlp in the GNN layer itself
        self.comb_norm = nn.LayerNorm(2 * d_pet)
        self.comb_mlp = nn.Sequential(
            nn.Linear(2 * d_pet, 2 * d_pet),
            nn.SiLU(),
            nn.Linear(2 * d_pet, d_pet),
        )


class ReadoutHead(nn.Module):
    """Single readout head."""
    def __init__(self, d_node: int, d_pet: int, d_head: int):
        super().__init__()
        self.node_head = nn.Sequential(
            nn.Linear(d_node, d_head),
            nn.SiLU(),
            nn.Linear(d_head, d_head),
            nn.SiLU(),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(d_pet, d_head),
            nn.SiLU(),
            nn.Linear(d_head, d_head),
            nn.SiLU(),
        )
        self.node_last = nn.Linear(d_head, 1)
        self.edge_last = nn.Linear(d_head, 1)
    
    def forward(
        self, node_feat: torch.Tensor, edge_feat: torch.Tensor, cutoffs_2d: torch.Tensor
    ) -> torch.Tensor:
        node_pred = self.node_last(self.node_head(node_feat))[:, 0]
        edge_pred = (self.edge_last(self.edge_head(edge_feat))[:, :, 0] * cutoffs_2d).sum(1)
        return node_pred + edge_pred


class UPET(nn.Module):
    def __init__(
        self,
        d_pet: int = 128,
        d_node: int = 512,
        d_head: int = 128,
        d_feedforward: int = 256,
        num_heads: int = 8,
        num_attention_layers: int = 1,
        num_gnn_layers: int = 2,
        cutoff: float = 7.5,
        cutoff_width: float = 0.5,
        cutoff_function: str = "Bump",
        normalization: str = "RMSNorm",
        activation: str = "SwiGLU",
        transformer_type: str = "PreLN",
        attention_temperature: float = 1.0,
        featurizer_type: str = "feedforward",
        num_species: int = 103,
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.cutoff_width = cutoff_width
        self.d_pet = d_pet
        self.d_node = d_node
        self.featurizer_type = featurizer_type
        self.num_gnn_layers = num_gnn_layers
        self.is_feedforward = featurizer_type == "feedforward"

        self.register_buffer("probes", torch.arange(0.5, cutoff, cutoff_width / 4))

        self.edge_embedder = nn.Embedding(num_species, d_pet)

        self.gnn_layers = nn.ModuleList(
            [
                GNNLayer(
                    d_pet,
                    d_node,
                    d_feedforward,
                    num_heads,
                    num_attention_layers,
                    normalization,
                    activation,
                    transformer_type,
                    attention_temperature,
                    num_species,
                    is_first=(i == 0),
                )
                for i in range(num_gnn_layers)
            ]
        )

        if self.is_feedforward:
            self.node_embedders = nn.ModuleList([nn.Embedding(num_species, d_node)])
            num_readouts = 1
        else:
            self.node_embedders = nn.ModuleList(
                [nn.Embedding(num_species, d_node) for _ in range(num_gnn_layers)]
            )
            num_readouts = num_gnn_layers

        self.readout_heads = nn.ModuleList(
            [ReadoutHead(d_node, d_pet, d_head) for _ in range(num_readouts)]
        )

    def forward(
        self,
        R_ij: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        species: torch.Tensor,
        reverse: torch.Tensor,
        pair_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_cutoffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        P = R_ij.shape[0]
        N = species.shape[0]
        n = P // N

        pair_mask_f = pair_mask.to(R_ij.dtype)
        atom_mask_f = atom_mask.to(R_ij.dtype)

        r_ij = torch.sqrt((R_ij**2).sum(-1) + 1e-15)

        cutoffs = cutoff_bump(r_ij, self.cutoff, self.cutoff_width) * pair_mask_f

        R_ij_2d = R_ij.reshape(N, n, 3)
        r_ij_2d = r_ij.reshape(N, n)
        cutoffs_2d = cutoffs.reshape(N, n)
        pair_mask_2d = pair_mask_f.reshape(N, n)
        neighbor_species = species[neighbors].reshape(N, n)

        central = torch.ones(N, 1, device=R_ij.device, dtype=R_ij.dtype)
        cutoffs_ext = torch.cat([central, cutoffs_2d], dim=1)
        mask_ext = torch.cat([atom_mask_f[:, None], pair_mask_2d], dim=1).to(torch.bool)
        cutoffs_ext = torch.where(mask_ext, cutoffs_ext, torch.zeros_like(cutoffs_ext))
        cutoff_matrix = cutoffs_ext[:, None, :].expand(N, 1 + n, -1)

        edge_msg = self.edge_embedder(neighbor_species) * pair_mask_2d[..., None]

        predictions = torch.zeros(N, device=species.device, dtype=R_ij.dtype)

        if self.is_feedforward:
            node_emb = self.node_embedders[0](species) * atom_mask_f[..., None]

            for gnn in self.gnn_layers:
                edge_geom = torch.cat([R_ij_2d, r_ij_2d[..., None]], dim=-1)
                edge_feat = gnn.edge_embed(edge_geom) * pair_mask_2d[..., None]

                if gnn.is_first:
                    tokens = gnn.compress(torch.cat([edge_feat, edge_msg], dim=-1))
                else:
                    neighbor_feat = gnn.neighbor_embed(neighbor_species) * pair_mask_2d[..., None]
                    tokens = gnn.compress(
                        torch.cat([edge_feat, neighbor_feat, edge_msg], dim=-1)
                    )

                node_tok = node_emb[:, None, :]
                for trans_layer in gnn.trans_layers:
                    node_tok, tokens = trans_layer(node_tok, tokens, cutoff_matrix)
                node_emb = node_tok[:, 0, :]

                tokens_flat = tokens.reshape(P, -1)
                reversed_flat = tokens_flat[reverse]
                reversed_2d = reversed_flat.reshape(N, n, -1)
                edge_msg = edge_msg + tokens + gnn.comb_mlp(gnn.comb_norm(torch.cat([tokens, reversed_2d], dim=-1)))

            # Single readout for feedforward
            predictions = self.readout_heads[0](node_emb, edge_msg, cutoffs_2d)

        else:  # residual
            readout_idx = 0
            for i, gnn in enumerate(self.gnn_layers):
                node_emb = self.node_embedders[i](species) * atom_mask_f[..., None]

                edge_geom = torch.cat([R_ij_2d, r_ij_2d[..., None]], dim=-1)
                edge_feat = gnn.edge_embed(edge_geom) * pair_mask_2d[..., None]

                if gnn.is_first:
                    tokens = gnn.compress(torch.cat([edge_feat, edge_msg], dim=-1))
                else:
                    neighbor_feat = gnn.neighbor_embed(neighbor_species) * pair_mask_2d[..., None]
                    tokens = gnn.compress(
                        torch.cat([edge_feat, neighbor_feat, edge_msg], dim=-1)
                    )

                node_tok = node_emb[:, None, :]
                for trans_layer in gnn.trans_layers:
                    node_tok, tokens = trans_layer(node_tok, tokens, cutoff_matrix)

                # Readout at each layer for residual
                predictions = predictions + self.readout_heads[readout_idx](
                    node_tok[:, 0, :], tokens, cutoffs_2d
                )
                readout_idx = readout_idx + 1

                tokens_flat = tokens.reshape(P, -1)
                reversed_flat = tokens_flat[reverse]
                reversed_2d = reversed_flat.reshape(N, n, -1)
                edge_msg = 0.5 * (edge_msg + reversed_2d)

        return predictions * atom_mask_f


def predict(
    model: UPET,
    batch: dict,
    compute_forces: bool = True,
    compute_stress: bool = False,
) -> dict:
    """Compute energy, forces, stress."""
    need_grad = compute_forces or compute_stress
    R_ij = batch["R_ij"]
    if need_grad:
        R_ij = R_ij.clone().requires_grad_(True)

    species = batch["species"]
    centers = batch["centers"]
    neighbors = batch["neighbors"]
    reverse = batch["reverse"]
    pair_mask = batch["pair_mask"]
    atom_mask = batch["atom_mask"]
    atom_to_struct = batch["atom_to_structure"]
    N = batch["num_atoms"]
    num_struct = batch["num_structures"]

    device = R_ij.device
    dtype = R_ij.dtype

    per_atom_energy = model(
        R_ij, centers, neighbors, species, reverse, pair_mask, atom_mask, None
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
