import torch
import os
import cv2
import imageio
import json

import numpy as np

from utils.geo_utils import compute_scale_and_shift, coord_trans_normal

# the code based on the GO-Surf

class ScannetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 dataset_dir: str,
                 device,
                 near: float = 0.01,
                 far: float = 5.0,
                 load=True,
                 ):
        super(ScannetDataset).__init__()

        self.device = device
        self.dataset_dir = dataset_dir

        self.args = args

        #self.rgb_dir = os.path.join(self.dataset_dir, 'train', 'frames', 'omni_full_color')
        self.rgb_dir = os.path.join(self.dataset_dir, 'train', 'rgb')
        self.depth_dir = os.path.join(self.dataset_dir, 'train', 'frames', 'omni_full_depth')
        self.gt_depth_dir = os.path.join(self.dataset_dir, 'train', 'depth')
        self.pose_dir = os.path.join(self.dataset_dir, 'transforms_train.json')
        self.normal_dir = os.path.join(self.dataset_dir, 'train', 'frames', 'omni_full_normal')
        self.rgb_pattern = '{:d}.jpg'
        self.depth_pattern = '{:d}.png'
        self.pose_pattern = '{:d}.txt'
        self.normal_pattern = '{:d}.png'

        self.target_rgb_dir = os.path.join(self.dataset_dir, 'test', 'rgb')
        self.target_depth_dir = os.path.join(self.dataset_dir, 'test', 'target_depth')
        self.target_pose_dir = os.path.join(self.dataset_dir, 'transforms_test.json')

        with open(self.pose_dir, 'r') as f:
            json_dict = json.load(f)

        with open(self.target_pose_dir, 'r') as f:
            json_dict_target = json.load(f)

        fx = json_dict['frames'][0]['fx']
        fy = json_dict['frames'][0]['fx']
        cx = json_dict['frames'][0]['cx']
        cy = json_dict['frames'][0]['cy']

        intri_rgb = np.asarray([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]]).astype(np.float32)

        intri_depth = np.asarray([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]]).astype(np.float32)

        """
        fx_t = json_dict_target['frames'][0]['fx']
        fy_t = json_dict_target['frames'][0]['fy']
        cx_t = json_dict_target['frames'][0]['cx']
        cy_t = json_dict_target['frames'][0]['cy']

        self.K_target = np.asarray([[fx_t, 0, cx_t],
                                    [0, fy_t, cy_t],
                                    [0, 0, 1]]).astype(np.float32)"""
        
        # due to crop there
        w = self.args.rendering.W
        h = self.args.rendering.H

        #intri_rgb[0,:] *= w / 624
        #intri_rgb[1,:] *= h / 468

        #intri_depth[0,:] *= w / 624
        #intri_depth[1,:] *= h / 468

        self.intrinsics_rgb = torch.from_numpy(intri_rgb)
        n_frames = len(os.listdir(self.rgb_dir))
        n_frames_target = len(os.listdir(self.target_rgb_dir))

        self.H = self.args.rendering.H + 2 * self.args.crop
        self.W = self.args.rendering.W + 2 * self.args.crop
           
        self.intri_depth = intri_depth
        self.intrinsics_depth = torch.from_numpy(intri_depth)
        self.near = near
        self.far = far
        self.frame_ids = list()
        self.load = load
            
        self.c2w_list = list()
        self.rgb_list = list()
        self.depth_list = list()
        self.K_list = list()
        self.scale_shift_list = list()
        self.normal_list = list()
        
        self.frame_ids_target = list()
        self.c2w_list_target = list()
        self.rgb_list_target = list()
        self.depth_list_target = list()
            
        for i in range(n_frames):

            c2w = np.asarray(json_dict['frames'][i]['transform_matrix']).astype(np.float32)
            c2w = torch.from_numpy(c2w)
            
            self.frame_ids.append(int(json_dict['frames'][i]['file_path'].split('/')[-1].split('.')[0]))
            self.c2w_list.append(c2w)

        for i in range(n_frames_target):
            c2w = np.asarray(json_dict_target['frames'][i]['transform_matrix']).astype(np.float32)
            c2w = torch.from_numpy(c2w)
            
            self.frame_ids_target.append(int(json_dict_target['frames'][i]['file_path'].split('/')[-1].split('.')[0]))
            self.c2w_list_target.append(c2w)

        if load:
            self.get_all_frames()

        # build normal again
        c2w_total = torch.cat([c2w.unsqueeze(0) for c2w in self.c2w_list], 0)
        new_normal = coord_trans_normal(self.normal_list.permute(0,2,3,1), c2w_total, mode='img')
        self.normal_list = new_normal.permute(0,3,1,2)
        
    def get_all_frames(self):
        
        self.pts_zero_idx = list()
        
        for i, frame_id in enumerate(self.frame_ids):
            rgb_path = os.path.join(self.rgb_dir, self.rgb_pattern.format(frame_id))
            depth_path = os.path.join(self.depth_dir, self.depth_pattern.format(frame_id))

            rgb = np.array(imageio.imread(rgb_path)).astype(np.float32)
            rgb = torch.as_tensor(rgb).permute(2, 0, 1)
            rgb /= 255.
            rgb= rgb[:,self.args.crop:self.H-self.args.crop, self.args.crop:self.W-self.args.crop]

            depth = 1 - cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)/255.
            depth_filtered = depth
            
            depth_filtered[depth == 0.] = 0.
            depth_filtered = np.nan_to_num(depth_filtered)
            depth = torch.from_numpy(depth_filtered)
            depth[depth < self.near] = 0.
            depth[depth > self.far] = 0.
            
            #depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), (self.H, self.W), mode='nearest').squeeze()
            depth = depth[self.args.crop:self.H-self.args.crop, self.args.crop:self.W-self.args.crop]
            
            normal_path = os.path.join(self.normal_dir, self.normal_pattern.format(frame_id))
            normal = cv2.imread(normal_path)
            normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
            normal = torch.from_numpy(normal).permute(2,0,1)
            
            normal = torch.nn.functional.interpolate(normal.unsqueeze(0), (self.H, self.W), mode='nearest').squeeze()
            normal = normal[:,self.args.crop:self.H-self.args.crop, self.args.crop:self.W-self.args.crop]
            self.normal_list.append(normal)
    
            self.rgb_list.append(rgb)
            self.K_list.append(self.intrinsics_depth)

            gt_depth = os.path.join(self.gt_depth_dir, self.depth_pattern.format(frame_id))
            gt_depth = cv2.imread(gt_depth, cv2.IMREAD_ANYDEPTH) / 1000.0
            gt_depth = torch.from_numpy(gt_depth).unsqueeze(0)
            
            gt_depth = torch.nn.functional.interpolate(gt_depth.unsqueeze(0), (384, 512), mode='nearest').squeeze(1)
            #gt_depth = torch.nn.functional.interpolate(gt_depth.unsqueeze(0), (self.H, self.W), mode='nearest').squeeze(1)
            gt_depth = gt_depth[:, self.args.crop:self.H-self.args.crop, self.args.crop:self.W-self.args.crop]
                
            valid_depth = gt_depth > 0

            # since reconstructed colmap has big error for each points,
            # we should build filter or consider KL divergence something.
            
            scale, shift = compute_scale_and_shift(depth.unsqueeze(0), gt_depth, valid_depth)
            scale_shift = torch.FloatTensor([scale, shift])
            self.scale_shift_list.append(scale_shift)

            depth = depth * scale + shift
            
            depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), (self.H, self.W), mode='nearest').squeeze()
            
            self.depth_list.append(depth)
                
        self.rgb_list = torch.stack(self.rgb_list, dim=0)
        self.depth_list = torch.stack(self.depth_list, dim=0)
        self.K_list = torch.stack(self.K_list, dim=0)
        self.scale_shift_list = torch.stack(self.scale_shift_list, dim=0)
        self.normal_list = torch.stack(self.normal_list, dim=0)
            
        for i, frame_id in enumerate(self.frame_ids_target):
            rgb_path = os.path.join(self.target_rgb_dir, self.rgb_pattern.format(frame_id))
            depth_path = os.path.join(self.target_depth_dir, self.depth_pattern.format(frame_id))

            rgb = np.array(imageio.imread(rgb_path)).astype(np.float32)
            rgb = torch.as_tensor(rgb).permute(2, 0, 1)
            rgb /= 255.

            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / 1000.0
            depth = torch.from_numpy(depth).unsqueeze(0)
    
            self.rgb_list_target.append(rgb)
            self.depth_list_target.append(depth)

        self.rgb_list_target = torch.stack(self.rgb_list_target, dim=0)
        self.depth_list_target = torch.stack(self.depth_list_target, dim=0)

    def __len__(self):
        return len(self.frame_ids)
