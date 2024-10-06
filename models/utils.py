import os
import torch

# the code was provided by GO-Surf

def compute_world_dims(bounds, voxel_sizes, n_levels, margin=0, device=torch.device("cpu")):
    coarsest_voxel_dims = ((bounds[:,1] - bounds[:,0] + margin*2) / voxel_sizes[-1])
    coarsest_voxel_dims = torch.ceil(coarsest_voxel_dims) + 1
    coarsest_voxel_dims = coarsest_voxel_dims.int()
    world_dims = (coarsest_voxel_dims - 1) * voxel_sizes[-1]
    
    # Center the model in within the grid
    volume_origin = bounds[:,0] - (world_dims - bounds[:,1] + bounds[:,0]) / 2
    
    # Multilevel dimensions
    voxel_dims = (coarsest_voxel_dims.view(1,-1).repeat(n_levels,1) - 1) * (voxel_sizes[-1] / torch.tensor(voxel_sizes).unsqueeze(-1)).int() + 1

    return world_dims.to(device), volume_origin.to(device), voxel_dims.to(device)
