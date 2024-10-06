import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from models.voxel_grid import FFSNeRF
from utils.geo_utils import coordinates

def init_geometry(args, optimizer, model, inv_s, logger, device):
    center = model.world_dims / 2. + model.volume_origin
    radius = model.world_dims.min() / 2.

    for iter_init_geo in tqdm(range(400)):
        optimizer.zero_grad()
        coords = coordinates(model.voxel_dims[1] - 1, device).float().t()
        pts = (coords + torch.rand_like(coords)) * args.voxel_sizes[1] + model.volume_origin

        sdf, *_ = model.get_sdf(pts.unsqueeze(1))
        sdf = sdf.squeeze(-1)
        target_sdf = radius - (center - pts).norm(dim=-1)
        loss = F.mse_loss(sdf, target_sdf)

        if iter_init_geo % 100 == 0:
            print_item = f'init_geo {iter_init_geo:03d}/500 | loss: {loss}'
            logger.info(print_item)
        
        if loss.item() < 1e-10:
            break

        loss.backward()
        optimizer.step()

    # build new optimizer
    param_list = list()
    param_list.append({"params": model.sdf_decoder.parameters(), "lr": args.lr.decoder})
    param_list.append({"params": model.rgb_decoder.parameters(), "lr": args.lr.decoder})
    param_list.append({"params": model.grid.parameters(), "lr": args.lr.features})
    param_list.append({"params": inv_s, "lr": args.lr.inv_s})
    optimizer = torch.optim.Adam(param_list)

    return optimizer, model
    
def build_optimizer_model(args, logger, device):

    model = FFSNeRF(args, device)

    inv_s = nn.parameter.Parameter(torch.tensor(args.inv_s, device=device))
    
    optimizer = torch.optim.Adam([{"params": model.sdf_decoder.parameters(), "lr": args.lr.decoder},
                                  {"params": model.rgb_decoder.parameters(), "lr": args.lr.decoder},
                                  {"params": model.grid.parameters(), "lr": args.lr.features},
                                  {"params": inv_s, "lr": args.lr.inv_s}])

    log_file_list = os.listdir(args.log_dir)

    ckpt_list = list()

    for fname in log_file_list:
        if 'ckpt' in fname:
            ckpt_list.append(fname)

    if len(ckpt_list) > 0:            
        last_ckpt_fname = sorted(ckpt_list)[-1]

        ckpt_path = os.path.join(
            args.log_dir,
            last_ckpt_fname
            )

        state_dict = torch.load(ckpt_path, map_location=device)

        inv_s = state_dict['inv_s']
        model.load_state_dict(state_dict['model'])
        start_iter = state_dict['iteration']
        optimizer.load_state_dict(state_dict['optimizer'])
    else:
        start_iter = 0
        optimizer, model = init_geometry(args,
                                         optimizer,
                                         model,
                                         inv_s,
                                         logger,
                                         device)
        
    return optimizer, model, inv_s, start_iter
