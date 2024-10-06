import os
import argparse
import numpy as np
import torch

from tqdm import tqdm
import open3d as o3d

from skimage.metrics import structural_similarity
import torch.nn.functional as F
import matplotlib.pyplot as plt

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)

def compute_rmse(prediction, target):
    return torch.sqrt((prediction - target).pow(2).mean())

# the height, weight and ground truth image should be changed here

def evaluation(args,
               model,
               dataset,
               inv_s,
               iiter,
               lpips_alex,
               logger):
    
    save_path = os.path.join(args.log_dir, 'viz_eval')
    os.makedirs(save_path, exist_ok=True)

    save_img_path = os.path.join(args.log_dir, 'viz_eval', f'eval_{iiter[0]}')
    os.makedirs(save_img_path, exist_ok=True)

    #fx, fy = dataset.K_target[0,0], dataset.K_target[1,1]
    #cx, cy = dataset.K_target[0,2], dataset.K_target[1,2]
    fx, fy = dataset.K_list[0, 0, 0], dataset.K_list[0, 1, 1]
    cx, cy = dataset.K_list[0, 0, 2], dataset.K_list[0, 1, 2]

    H = args.rendering.H
    W = args.rendering.W
    
    u, v = torch.arange(W), torch.arange(H)
    v, u = torch.meshgrid(v, u)

    # save target images
    rgb_total_list = list()
    depth_total_list = list()
    normal_total_list = list()
    cams_total_list = list()

    length = len(dataset.c2w_list_target)
    
    # evaluation of psnr
    for i in tqdm(range(length)):
        save_img_fname = os.path.join(args.log_dir, 'viz_eval', f'eval_{iiter[0]}', f'index{i:04d}.png')
        
        if args.rendering.inverse_y:
            rays_d_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1).cuda()
        else:  # OpenGL
            rays_d_cam = torch.stack([(u - cx) / fx, -(v - cy) / fy, -torch.ones_like(v)], dim=-1).cuda()

        c2w = dataset.c2w_list_target[i].cuda()

        rays_o = c2w[:3,3].unsqueeze(0).repeat(H*W, 1).cuda()
        rays_d = (c2w[:3, :3] @ rays_d_cam[...,None]).reshape(H*W, -1).cuda()

        size = 4096

        rgb_list = list()
        depth_list = list()
        normal_list = list()
        uncertainty_list = list()
        mask_list = list()
        
        for rays_o_split, rays_d_split in zip(torch.split(rays_o, size), torch.split(rays_d, size)):
            dummy_gt_dict = {
                'rgb': torch.ones(rays_o_split.shape[0], 3).cuda().float(),
                'depth': torch.ones(rays_o_split.shape[0]).cuda().float(),
                'normal': torch.ones(rays_o_split.shape[0], 3).cuda().float()
                }
            
            ret = model(
                rays_o_split.cuda(),
                rays_d_split.cuda(),
                dummy_gt_dict,
                inv_s=torch.exp(10. * inv_s),
                iter=iiter,
                test=True
            )

            rgb_list.append(ret['rgb'].cpu().detach())
            depth_list.append(ret['depth'].cpu().detach())
            normal_list.append(ret['normal'].cpu().detach())

        rgb = torch.cat(rgb_list, 0).reshape(H, W, 3).clip(0,1).unsqueeze(0)
        depth = torch.cat(depth_list, 0).reshape(H, W).unsqueeze(0)
        normal = torch.cat(normal_list, 0).reshape(H, W, 3).unsqueeze(0)
        normal = normal + 1.
        normal = normal / 2
        normal = torch.clip(normal, 0, 1)

        plt.figure(figsize=(20,12))
        plt.tight_layout()
        plt.subplot(231)
        plt.imshow(rgb.squeeze().cpu().numpy())
        plt.subplot(232)
        plt.imshow(depth.squeeze().cpu().numpy())
        plt.colorbar()
        plt.subplot(233)
        plt.imshow(normal.squeeze().cpu().numpy())
        plt.subplot(234)
        plt.imshow(dataset.rgb_list_target[i].permute(1,2,0).clip(0,1).cpu().numpy())
        plt.subplot(235)
        plt.imshow(dataset.depth_list_target[i].squeeze().cpu().numpy())
        plt.colorbar()
        plt.savefig(save_img_fname)
        plt.clf()

        rgb_total_list.append(rgb)
        depth_total_list.append(depth)
        normal_total_list.append(normal)
        cams_total_list.append(c2w)

    # calculate psnr here
    pred_rgb = torch.cat(rgb_total_list).permute(0,3,1,2) # [B, C, H, W]
    gt_rgb = dataset.rgb_list_target #_origin

    psnr_list = list()
    lpips_list = list()
    ssim_list = list()
    
    for i in range(pred_rgb.shape[0]):
        rgb_loss = F.mse_loss(pred_rgb[i], gt_rgb[i])
        psnr = mse2psnr(rgb_loss).item()
        psnr_list.append(psnr)

    with torch.no_grad():
        for i in range(length):
            temp_pred = pred_rgb[i]
            temp_gt = gt_rgb[i]

            lpips_temp = lpips_alex(temp_pred, temp_gt, normalize=True)[0].item()
            ssim_temp = structural_similarity(temp_pred.cpu().numpy(), temp_gt.cpu().numpy(), data_range=1., channel_axis=0)
            
            lpips_list.append(lpips_temp)
            ssim_list.append(ssim_temp)

        
    logger.info('-------------------------Eval--------------------------')
    
    for i in range(length):
        logger.info(f'iter {iiter[0]} || img{i}: PSNR {psnr_list[i]:.3f} || lpips {lpips_list[i]:.3f} || ssim {ssim_list[i]:.3f}')

    psnr = sum(psnr_list) / len(psnr_list)
    lpips = sum(lpips_list) / len(lpips_list)
    ssim = sum(ssim_list) / len(ssim_list)

    pred_depth = torch.cat(depth_total_list)
    target_depth = dataset.depth_list_target.squeeze()

    depth_mask = target_depth > 0
    
    rmse = compute_rmse(pred_depth[depth_mask], target_depth[depth_mask])
    
    logger.info(f'iter {iiter[0]} || Test: PSNR {psnr:.3f} || lpips {lpips:.3f} || ssim {ssim:.3f} || rmse {rmse:.3f}')
    
    print('Save path: {}'.format(args.log_dir))

    return psnr, lpips, ssim, rmse
