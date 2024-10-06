import torch

# the code was provided by MiDAS

def coordinates(voxel_dim, device: torch.device):
    nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))

def compute_scale_and_shift(prediction, target, mask): 
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]

    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2)) 
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b 
    x_0 = torch.zeros_like(b_0) 
    x_1 = torch.zeros_like(b_1) 
    det = a_00 * a_11 - a_01 * a_01 
    valid = det.nonzero() 
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / (det[valid] + 1e-6) 
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / (det[valid] + 1e-6)
    return x_0, x_1

def norm_norm(normal):
    normal = normal / 255. - 0.5
    normal = normal * 2.
    return normal

# coordinate conversion for normal
def coord_trans_normal(normal, pose, mode='pred'):
    """
    Args:
        normal: normal map [B, H, W, 3] or [B, -, 3]
        pose  : camera pose [B, 4, 4]
        mode  : 'pred' ==> predicted (would be in range (-1, 1) else 0~255
    """
    img_shape = False

    if mode=='img':
        normal = norm_norm(normal)
    
    if len(normal.shape) == 4:
        b, h, w, _ = normal.shape
        normal = normal.reshape(b, -1,3)
        img_shape = True

    #pose = pose.clone().detach()
        
    pose[:,:3,1] = -pose[:,:3,1]
    pose[:,:3,2] = -pose[:,:3,2]
        
    normal_map = pose[:,:3,:3].bmm(normal.permute(0,2,1)).permute(0,2,1).reshape(b, h, w, -1)
    normal_map_norm = torch.linalg.norm(normal_map, axis=-1)

    normal_map = normal_map / normal_map_norm.unsqueeze(-1)
    
    pose[:,:3,1] = -pose[:,:3,1]
    pose[:,:3,2] = -pose[:,:3,2]
    
    return normal_map
