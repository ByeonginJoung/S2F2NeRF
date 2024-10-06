import torch

import numpy as np

import torch.nn.functional as F

from models.decoders import GeometryDecoder, RadianceDecoder
from models.utils import compute_world_dims
from models.multi_grid import MultiGrid

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)

class FFSNeRF(torch.nn.Module):
    def __init__(self,
                 args,
                 device):
        super(FFSNeRF, self).__init__()

        self.args = args
        
        world_dims, volume_origin, voxel_dims = compute_world_dims(torch.FloatTensor(args.scene_sizes),
                                                                   args.voxel_sizes,
                                                                   len(args.voxel_sizes),
                                                                   margin=0.1,
                                                                   device=device)
        
        self.world_dims = world_dims
        self.volume_origin = volume_origin
        self.voxel_dims = voxel_dims
        
        grid_dim = (torch.tensor(args.sdf_feature_dim) + torch.tensor(args.rgb_feature_dim)).tolist()
        
        self.grid = MultiGrid(voxel_dims, grid_dim).to(device)
        
        self.sdf_decoder = GeometryDecoder(
            W = args.geometry['W'],
            D = args.geometry['D'],
            skips = args.geometry['skips'],
            input_feat_dim = sum(args.sdf_feature_dim),
            n_freq = args.geometry['n_freq'],
            args = args
        ).to(device)
        
        self.rgb_decoder = RadianceDecoder(
            W = args.radiance['W'],
            D = args.radiance['D'],
            skips = args.radiance['skips'],
            input_feat_dim = sum(args.rgb_feature_dim),
            n_freq = args.radiance['n_freq'],
            args = args
        ).to(device)

    def return_test_results(self, pred_dict):
        rgb = pred_dict['rgb']
        depth = pred_dict['depth']
        normal = pred_dict['normal']

        return {
            'rgb': rgb,
            'depth': depth,
            'normal': normal
            }
        
    def return_train_results(self, pred_dict, gt_dict):
        color_loss = F.mse_loss(pred_dict['rgb'], gt_dict['rgb'])
        psnr = mse2psnr(color_loss)
        valid_depth_mask = gt_dict['depth'] > 0.
        valid_depth_mask = valid_depth_mask & (gt_dict['depth'] < self.args.far)

        if valid_depth_mask.int().sum() == 0:
            depth_loss = torch.FloatTensor([0.]).cuda()
        else:
            depth_loss = F.mse_loss(pred_dict['depth'][valid_depth_mask],
                                    gt_dict['depth'][valid_depth_mask])

        normal_loss = F.mse_loss(pred_dict['normal'], gt_dict['normal'])

        asdf_loss = pred_dict['asdf_loss']
        eikonal_loss = pred_dict['eikonal_loss']
        
        return {
            'color_loss': color_loss,
            'depth_loss': depth_loss,
            'normal_loss': normal_loss,
            'asdf_loss': asdf_loss,
            'eikonal_loss': eikonal_loss,
            'psnr': psnr
            }

    def get_sdf(self,
                query_pts):
        
        pts_norm = 2. * (query_pts - self.volume_origin[None, None, :]) / self.world_dims[None, None, :] - 1.
        mask = (pts_norm.abs() <= 1.).all(dim=-1)
        pts_norm = pts_norm[mask].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        mlvl_feats = self.grid(pts_norm[...,[2,1,0]], concat=False)
        sdf_feats = list(map(lambda feat_pts, rgb_dim: feat_pts[:,:-rgb_dim,...] \
                             if rgb_dim > 0 else feat_pts,
                             mlvl_feats, self.args.rgb_feature_dim))
        
        sdf_feats = torch.cat(sdf_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()

        rgb_feats = map(lambda feat_pts, rgb_dim: feat_pts[:,-rgb_dim:,...] \
                        if rgb_dim > 0 else None,
                        mlvl_feats, self.args.rgb_feature_dim)

        rgb_feats = list(filter(lambda x: x is not None, rgb_feats))
        
        rgb_feats = torch.cat(rgb_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()

        rgb_feats_unmasked = torch.zeros(list(mask.shape) + [sum(self.args.rgb_feature_dim)], device=pts_norm.device)
        
        rgb_feats_unmasked[mask] = rgb_feats
        
        sdf_feats = torch.cat([sdf_feats, pts_norm.squeeze()], -1)

        raw = self.sdf_decoder(sdf_feats)
        sdf = torch.zeros_like(mask, dtype=raw.dtype)
        sdf[mask] = raw.squeeze(-1)
        
        return sdf, rgb_feats_unmasked, mask

    def compute_grads(self,
                      pred_sdf,
                      query_pts):
        
        grad, = torch.autograd.grad(
            [pred_sdf],
            [query_pts],
            [torch.ones_like(pred_sdf)],
            create_graph=True
            )

        return grad
    
    def render(self,
               rays_o,
               rays_d,
               gt_dict,
               inv_s,
               iter,
               test):
        
        n_rays = rays_o.shape[0]
        z_vals = torch.linspace(self.args.near, self.args.far, self.args.n_samples).to(rays_o)
        z_vals = z_vals[None, :].repeat(n_rays, 1)  # [n_rays, n_samples]
        sample_dist = (self.args.far - self.args.near) / self.args.n_samples

        z_vals += torch.rand_like(z_vals) * sample_dist
       
        n_importance_steps = self.args.n_importance // self.args.n_importance_step_size
        
        with torch.no_grad():
            for step in range(n_importance_steps):
                query_points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf, *_ = self.get_sdf(query_points)

                prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
                prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
                mid_sdf = (prev_sdf + next_sdf) * 0.5
                cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

                prev_cos_val = torch.cat([torch.zeros([n_rays, 1], device=z_vals.device), cos_val[:, :-1]], dim=-1)
                cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
                cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
                cos_val = cos_val.clip(-1e3, 0.0)

                dists = next_z_vals - prev_z_vals
                weights, _ = neus_weights(mid_sdf, dists, torch.tensor(64. * 2 ** step, device=mid_sdf.device), cos_val)
                z_samples = sample_pdf(z_vals, weights, self.args.n_importance_step_size, det=True).detach()
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
                
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.ones_like(gt_dict['depth']).unsqueeze(-1) * sample_dist], dim=1)

        z_vals_mid = z_vals + dists * 0.5
        view_dirs = F.normalize(rays_d, dim=-1)[:, None, :].repeat(1, self.args.n_samples + self.args.n_importance, 1)
        query_points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_mid[..., :, None]
        query_points = query_points.requires_grad_(True)

        sdf, rgb_feat, world_bound_mask = self.get_sdf(query_points)
        grads = self.compute_grads(sdf, query_points)

        rgb_feat = [rgb_feat]
    
        rgb_feat.append(view_dirs)
        rgb_feat.append(2. * (query_points - self.volume_origin) / self.world_dims - 1.)
            
        rgb = torch.sigmoid(self.rgb_decoder(torch.cat(rgb_feat, dim=-1)))
        
        cos_val = (view_dirs * grads).sum(-1)
        # cos_val = -F.relu(-cos_val)
        cos_anneal_ratio = min(iter[0] / iter[1], 1.)
        cos_val = -(F.relu(-cos_val * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                    F.relu(-cos_val) * cos_anneal_ratio)
        weights, alpha = neus_weights(sdf, dists, inv_s, cos_val)
        weights[~world_bound_mask] = 0.

        rendered_rgb = torch.sum(weights[..., None] * rgb, dim=-2)
        rendered_depth = torch.sum(weights * z_vals_mid, dim=-1)
        rendered_grads = torch.sum(weights[...,None] * grads, dim=-2)
        tmp = (z_vals_mid - rendered_depth.unsqueeze(-1))
        rendered_uncertainty = torch.sum(weights * tmp * tmp, dim=-1)
        rendered_grads = rendered_grads / rendered_grads.norm(dim=-1, keepdim=True)

        eikonal_weights = sdf[world_bound_mask].detach().abs() + 1e-2
        
        # eikonal grads
        ret_val_grads = grads.norm(dim=-1)[world_bound_mask].clone().detach()
        
        if self.args.use_eikonal_only:
            eikonal_loss = torch.square(grads.norm(dim=-1)[world_bound_mask] - 1.).mean()
            asdf_loss = torch.tensor(0.0, device=rays_o.device)            
        else:
            eikonal_loss = (torch.square(grads.norm(dim=-1)[world_bound_mask] - 1.) \
                            * eikonal_weights).sum() / eikonal_weights.sum()
            init_truncation = self.args.asdf_loss.growing_trunc_init_val
            last_trunc_iter = self.args.asdf_loss.growing_trunc_last_iter
            last_trunc_val = self.args.asdf_loss.growing_trunc_last_val

            bound = rendered_depth[:,None] - z_vals_mid
            new_truncation = max((last_trunc_iter - iter[0]) / last_trunc_iter * init_truncation, last_trunc_val)
            
            anneal_mask = (bound.abs() <= new_truncation)
            if anneal_mask.int().sum() == 0:
                asdf_loss = torch.tensor(0.0, device=rays_o.device)
            else:
                asdf_loss = (torch.abs(sdf - bound))[anneal_mask].mean() / 1.5

        depth_mask = (gt_dict['depth'] > 0.) & (gt_dict['normal'] != 0.).any(dim=-1)
        query_points = rays_o[depth_mask] + rays_d[depth_mask] * gt_dict['depth'][depth_mask, None]
        query_points = query_points.requires_grad_(True)
        sdf, *_ = self.get_sdf(query_points)
        
        normals = rendered_grads
        normal_loss = F.mse_loss(normals, gt_dict['normal'])
        
        if test:
            return {
                'rgb': rendered_rgb.detach(),
                'depth': rendered_depth.detach(),
                'normal': rendered_grads.detach()
               }
        else: # for train            
            return {
                'rgb': rendered_rgb,
                'depth': rendered_depth,
                'normal': rendered_grads,
                'asdf_loss': asdf_loss,
                'eikonal_loss': eikonal_loss,
                'normal_loss': normal_loss
                }
            
    def forward(self,
                rays_o,
                rays_d,
                gt_dict,
                inv_s,
                iter = 0,
                test=False):

        pred_dict = self.render(rays_o,
                                rays_d,
                                gt_dict,
                                inv_s,
                                iter,
                                test)
        
        if test:
            return self.return_test_results(pred_dict)
        else:
            return self.return_train_results(pred_dict, gt_dict)

def neus_weights(sdf, dists, inv_s, cos_val, z_vals=None):    
    estimated_next_sdf = sdf + cos_val * dists * 0.5
    estimated_prev_sdf = sdf - cos_val * dists * 0.5
    
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
    weights = alpha * torch.cumprod(torch.cat([torch.ones([sdf.shape[0], 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    
    if z_vals is not None:
        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0., torch.ones_like(signs), torch.zeros_like(signs))
        # This will only return the first zero-crossing
        inds = torch.argmax(mask, dim=1, keepdim=True)
        z_surf = torch.gather(z_vals, 1, inds)
        return weights, z_surf
    
    return weights, alpha

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    # device = weights.get_device()
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / N_importance, 1. - 0.5 / N_importance, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
