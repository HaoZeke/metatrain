import torch
from typing import List, Optional
from metatomic.torch import System


def systems_to_upet_batch(
    systems: List[System], 
    cutoff: float,
    species_to_index: dict,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    cutoff_width: float = 0.5,
    num_neighbors_adaptive: int = None,
):
    """
    Converts a list of metatomic Systems into the rectangular batch format required by UPET.
    """
    # Use provided dtype or infer from first system
    if dtype is None:
        dtype = systems[0].positions.dtype
    
    # First pass: collect all data and find max neighbors
    all_system_data = []
    total_atoms = 0
    max_neighbors = 0
    
    for i, system in enumerate(systems):
        n_atoms = len(system)
        
        # Get neighbor list - known_neighbor_lists() is a METHOD
        known_nls = system.known_neighbor_lists()
        if len(known_nls) == 0:
            raise ValueError(
                f"System {i} has no neighbor lists. "
                "Make sure to call get_system_with_neighbor_lists() first."
            )
        options = known_nls[0]
        nl = system.get_neighbor_list(options)
        
        # nl is a TensorBlock with samples [pair_idx, first_atom, second_atom]
        samples = nl.samples
        sample_values = samples.values
        local_centers = sample_values[:, 1]
        local_neighbors = sample_values[:, 2]
        
        # Displacement vectors
        vectors = nl.values.squeeze(-1)  # [n_pairs, 3]
        
        # Count neighbors per atom
        if len(local_centers) > 0:
            neighbor_counts = torch.bincount(local_centers.to(torch.long), minlength=n_atoms)
            local_max = int(neighbor_counts.max().item())
        else:
            neighbor_counts = torch.zeros(n_atoms, dtype=torch.long, device=local_centers.device)
            local_max = 0
        max_neighbors = max(max_neighbors, local_max)
        
        all_system_data.append({
            'n_atoms': n_atoms,
            'species': system.types,
            'local_centers': local_centers,
            'local_neighbors': local_neighbors,
            'vectors': vectors,
            'neighbor_counts': neighbor_counts,
            'atom_offset': total_atoms,
        })
        total_atoms += n_atoms
    
    max_neighbors = max(max_neighbors, 1)
    N = total_atoms
    n = max_neighbors
    P = N * n
    
    # Allocate with correct dtype
    R_ij = torch.zeros(P, 3, device=device, dtype=dtype)
    centers = torch.zeros(P, device=device, dtype=torch.long)
    neighbors = torch.zeros(P, device=device, dtype=torch.long)
    pair_mask = torch.zeros(P, device=device, dtype=torch.bool)
    species = torch.zeros(N, device=device, dtype=torch.long)
    atom_mask = torch.zeros(N, device=device, dtype=torch.bool)
    atom_to_structure = torch.zeros(N, device=device, dtype=torch.long)
    
    # Fill in data
    for sys_idx, data in enumerate(all_system_data):
        n_atoms = data['n_atoms']
        offset = data['atom_offset']
        sys_species = data['species']
        
        for local_idx in range(n_atoms):
            global_idx = offset + local_idx
            z = int(sys_species[local_idx].item())
            species[global_idx] = species_to_index.get(z, 0)
            atom_mask[global_idx] = True
            atom_to_structure[global_idx] = sys_idx
        
        local_centers = data['local_centers']
        local_neighbors = data['local_neighbors']
        vectors = data['vectors']
        
        for local_atom in range(n_atoms):
            global_atom = offset + local_atom
            atom_pair_start = global_atom * n
            
            mask = (local_centers == local_atom)
            atom_neighbors = local_neighbors[mask]
            atom_vectors = vectors[mask]
            num_neigh = len(atom_neighbors)
            
            for slot_idx in range(num_neigh):
                flat_idx = atom_pair_start + slot_idx
                R_ij[flat_idx] = atom_vectors[slot_idx].to(dtype)
                centers[flat_idx] = global_atom
                neighbors[flat_idx] = offset + int(atom_neighbors[slot_idx].item())
                pair_mask[flat_idx] = True
            
            for slot_idx in range(num_neigh, n):
                flat_idx = atom_pair_start + slot_idx
                centers[flat_idx] = global_atom
                neighbors[flat_idx] = global_atom

    reverse = _compute_reverse_indices(centers, neighbors, pair_mask, N, n, device)
    
    return {
        "R_ij": R_ij,
        "centers": centers,
        "neighbors": neighbors,
        "species": species,
        "reverse": reverse,
        "pair_mask": pair_mask,
        "atom_mask": atom_mask,
        "atom_to_structure": atom_to_structure,
        "num_atoms": N,
        "num_structures": len(systems),
        "cutoff": cutoff,
        "cutoff_width": cutoff_width,
        "num_neighbors_adaptive": num_neighbors_adaptive,
    }


def _compute_reverse_indices(centers, neighbors, pair_mask, N, n, device):
    """Compute reverse indices for message passing."""
    P = len(centers)
    reverse = torch.arange(P, device=device, dtype=torch.long)
    
    valid_indices = torch.where(pair_mask)[0]
    if len(valid_indices) == 0:
        return reverse
    
    valid_centers = centers[valid_indices]
    valid_neighbors = neighbors[valid_indices]
    
    Big = N + 1
    fwd_hash = valid_centers * Big + valid_neighbors
    bwd_hash = valid_neighbors * Big + valid_centers
    
    sorted_fwd, perm_fwd = torch.sort(fwd_hash)
    sorted_bwd, perm_bwd = torch.sort(bwd_hash)
    
    orig_fwd = valid_indices[perm_fwd]
    orig_bwd = valid_indices[perm_bwd]
    
    reverse[orig_fwd] = orig_bwd
    
    return reverse
