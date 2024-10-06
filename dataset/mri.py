import os
import numpy as np
import torch
import clip

from torch.utils.data import Dataset
from glob import glob
import scipy.io

class MRI(Dataset):
    def __init__(self,
                 args,
                 ):
        
        data_dir = os.path.join(args.data_path, args.name_scene)
        data_3d = os.path.join(data_dir, '3d', 'recon')

        fnames = glob(os.path.join(data_3d, '*'))
        
        text_list = list()
        self.data_list = list()

        model, preprocess = clip.load("ViT-B/32", device='cuda')

        for idx, fname in enumerate(fnames):
            text = fname.split('/')[-1].split('_')[3]
            
            if text not in text_list:              
                text_list.append(text)

                data0_path = fnames[idx-1]
                data1_path = fnames[idx]

                self.data_list.append([data0_path, data1_path])

        text_list = clip.tokenize(text_list).cuda()
        
        with torch.no_grad():
            self.text_features = model.encode_text(text_list).cpu().detach()

        del model
        del preprocess
        del text_list
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.text_features)

    def __getitem__(self, idx):

        text_feat = self.text_features[idx]
        data_path = self.data_list[idx]

        data_list = list()
        
        for data_p in data_path:
            data = np.abs(scipy.io.loadmat(data_p)['recon'])
            # normalize here
            data_min = data.min()
            data_max = data.max()
            data = (data - data_min) / (data_max - data_min)
            data_list.append(torch.from_numpy(data))

        data = torch.stack(data_list)
        
        return text_feat, data
