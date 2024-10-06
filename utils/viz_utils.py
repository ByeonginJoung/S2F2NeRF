import os
import torch

import numpy as np

import matplotlib.pyplot as plt

def visualization(args,
                  model,
                  dataset,
                  index,
                  iters,
                  inv_s,
                  iiters=[0,1]):
    
    save_path = os.path.join(args.log_dir, 'viz_imgs')
    os.makedirs(save_path, exist_ok=True)

    save_path = os.path.join(args.log_dir, 'viz_imgs', f'iters{iters}')
    os.makedirs(save_path, exist_ok=True)

    # get points
    fx, fy = dataset.K_list[index, 0, 0], dataset.K_list[index, 1, 1]
    cx, cy = dataset.K_list[index, 0, 2], dataset.K_list[index, 1, 2]

    length = len(dataset.c2w_list)
    for index in range(length):

        save_img_path = os.path.join(args.log_dir, 'viz_imgs', f'iters{iters}', f'recon_index{index:04d}_{iters}.png')

        gt_depth = dataset.depth_list[index]
        gt_rgb = dataset.rgb_list[index].permute(1,2,0)
        gt_normal = dataset.normal_list[index].permute(1,2,0).contiguous().clone()

        H, W = gt_depth.shape

        u, v = torch.arange(gt_depth.shape[1]), torch.arange(gt_depth.shape[0])

        v, u = torch.meshgrid(v, u)

        if args.rendering.inverse_y:
            rays_d_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(v)], dim=-1)
        else:  # OpenGL
            rays_d_cam = torch.stack([(u - cx) / fx, -(v - cy) / fy, -torch.ones_like(v)], dim=-1)
            
        c2w = dataset.c2w_list[index]

        rays_o = c2w[:3,3].unsqueeze(0).repeat(gt_depth.shape[0] * gt_depth.shape[1], 1).cuda()
        rays_d = (c2w[:3, :3] @ rays_d_cam[...,None]).reshape(gt_depth.shape[0] * gt_depth.shape[1], -1).cuda()

        size = 4096

        rgb_list = list()
        depth_list = list()
        normal_list = list()
        uncertainty_list = list()
        mask_list = list()

        gt_depth_view = gt_depth.reshape(H*W)
        gt_normal_view = gt_normal.reshape(H*W, -1)
        
        for rays_o_split, rays_d_split, depth_split, normal_split in zip(torch.split(rays_o, size),
                                                                         torch.split(rays_d, size),
                                                                         torch.split(gt_depth_view, size),
                                                                         torch.split(gt_normal_view, size)):

            gt_dict = {
                'rgb': gt_rgb.view(-1,3)[:rays_o_split.shape[0]].cuda(),
                'depth': depth_split.cuda().float(),
                'normal': normal_split.cuda().float()
                }
            
            ret = model(
                rays_o_split.cuda(),
                rays_d_split.cuda(),
                gt_dict,
                inv_s=torch.exp(10. * inv_s),
                iter=iiters,
                test=True
            )

            rgb_list.append(ret['rgb'].cpu().detach())
            depth_list.append(ret['depth'].cpu().detach())
            normal_list.append(ret['normal'].cpu().detach())

        gt_normal += 1.0
        gt_normal /= 2.0
        gt_normal = gt_normal.clip(0,1)

        rgb = torch.cat(rgb_list, 0).reshape(H, W, 3).numpy()
        rgb = np.clip(rgb ,0,1)
        depth = torch.cat(depth_list, 0).reshape(H, W).numpy()
        normal = torch.cat(normal_list, 0).reshape(H, W, 3).numpy()
        normal = normal + 1.
        normal = normal / 2
        normal = np.clip(normal, 0, 1)
        
        plt.figure(figsize=(30,12))
        plt.tight_layout()
        plt.subplot(231)
        plt.imshow(rgb)
        plt.subplot(232)
        plt.imshow(depth)
        plt.colorbar()
        plt.subplot(233)
        plt.imshow(normal)
        plt.subplot(234)
        plt.imshow(gt_rgb.cpu().numpy())
        plt.subplot(235)
        plt.imshow(gt_depth.cpu().numpy())
        plt.colorbar()
        plt.subplot(236)
        plt.imshow(gt_normal.cpu().numpy())
        plt.savefig(save_img_path)
        plt.clf()
