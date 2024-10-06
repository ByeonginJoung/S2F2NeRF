import os
import torch

import torch.nn.functional as F

from tqdm import trange
from lpips import LPIPS

#from dataset.scannet_dataset import ScannetDataset
from dataset.mri import MRI
from trainer.utils import build_optimizer_model

from utils.seed import set_seed
from utils.viz_utils import visualization
from utils.evaluator import evaluation

def run_trainer(args, logger):

    if args.iterations >= args.iter_eval:
        lpips_alex = LPIPS()
        metrics_list = list()
    else:
        metrics_list = None
        
    set_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    # load dataset
    dataset = MRI(args)

    import pdb; pdb.set_trace()
    
    poses_mat_init = torch.stack(dataset.c2w_list, dim=0).to(device)
    
    # set optimizer
    optimizer, model, inv_s, start_iter = build_optimizer_model(args, logger, device)
    
    ray_indices = torch.randperm(len(dataset) * dataset.H * dataset.W)

    img_stride = dataset.H * dataset.W
    n_batches = ray_indices.shape[0] // args.batch_size

    for iteration in trange(start_iter + 1, args.iterations + 1):
        batch_idx = iteration % n_batches
        ray_ids = ray_indices[(batch_idx * args.batch_size):((batch_idx + 1) * args.batch_size)]
        
        frame_id = ray_ids.div(img_stride, rounding_mode='floor')
        v = (ray_ids % img_stride).div(dataset.W, rounding_mode='floor')
        u = ray_ids % img_stride % dataset.W

        depth = dataset.depth_list[frame_id, v, u].to(device, non_blocking=True).float()
        rgb = dataset.rgb_list[frame_id, :, v, u].to(device, non_blocking=True)
        normal = dataset.normal_list[frame_id, :, v, u].to(device, non_blocking=True)

        fx, fy = dataset.K_list[frame_id, 0, 0], dataset.K_list[frame_id, 1, 1]
        cx, cy = dataset.K_list[frame_id, 0, 2], dataset.K_list[frame_id, 1, 2]
        
        if args.rendering.inverse_y:
            rays_d_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(fx)], dim=-1).to(device)
        else:  # OpenGL
            rays_d_cam = torch.stack([(u - cx) / fx, -(v - cy) / fy, -torch.ones_like(fy)], dim=-1).to(device)
            
        c2w = poses_mat_init[frame_id]
    
        rays_o = c2w[:,:3,3]
        rays_d = torch.bmm(c2w[:, :3, :3], rays_d_cam[..., None]).squeeze()

        gt_label_dict = {
            'rgb': rgb,
            'depth': depth,
            'normal': normal
            }

        ret = model(
            rays_o,
            rays_d,
            gt_label_dict,
            inv_s=torch.exp(10. * inv_s),
            iter=[iteration, args.iterations]
        )

        loss = args.color_weight * ret["color_loss"] +\
               args.depth_weight * ret["depth_loss"] +\
               args.asdf_weight * ret["asdf_loss"]  +\
               args.eikonal_weight * ret["eikonal_loss"] +\
               args.normal_weight * ret['normal_loss']
            
        torch.nn.utils.clip_grad_norm_(model.grid.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(model.rgb_decoder.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(model.sdf_decoder.parameters(), 1.)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iteration % args.iter_print == 0:

            init_viz = '\nIter: {} | '.format(iteration)
            psnr_viz = 'pnsr: {:4f} | '.format(ret['psnr'].item())
            color_viz = 'color: {:4f} | '.format(ret['color_loss'].item())
            depth_viz = 'depth: {:4f} | '.format(ret['depth_loss'].item())
            asdf_viz = 'asdf: {:4f} | '.format(ret['asdf_loss'].item())
            eik_viz = 'eikonal: {:4f}'.format(ret['eikonal_loss'].item())

            print_item = init_viz + psnr_viz + color_viz + depth_viz + asdf_viz + eik_viz
            logger.info(print_item)

        if iteration % args.iter_viz == 0:
            visualization(args,
                          model,
                          dataset,
                          index=0,
                          iters=iteration,
                          inv_s=inv_s,
                          iiters = [iteration, args.iterations])
            logger.info(f'Eval {args.name_scene} {args.name_log}')
            
        if iteration % args.iter_eval == 0:
            metrics = evaluation(args,
                                 model,
                                 dataset,
                                 inv_s,
                                 [iteration, args.iterations],
                                 lpips_alex,
                                 logger)
            metrics_list.append(metrics)

        # Save checkpoint
        
        if iteration % args.iter_save == 0:
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'iteration': iteration,
                     'inv_s': inv_s,
                     'metrics': metrics_list}
            torch.save(state, os.path.join(args.log_dir, "ckpt_{}".format(iteration)))

    if args.iterations >= args.iter_eval:            
        for idx, metric in enumerate(metrics_list):
            _iter = (idx + 1) * args.iter_eval

            init_viz = f'iter [{_iter:05d}/{args.iterations:05d}] '
            psnr_viz = f'psnr {metric[0]:.3f} '
            lpips_viz = f'lpips {metric[1]:.3f} '
            ssim_viz = f'ssim {metric[2]:.3f} '
            rmse_viz = f'rmse {metric[3].item():.3f}'

            print_item = init_viz + psnr_viz + lpips_viz + ssim_viz + rmse_viz
            logger.info(print_item)
