from easydict import EasyDict as edict

def get_args():
    args_edict = edict()
    args_edict.dataset_dir = '/ssd1tb_00/byeonginjoung/dataset/ddpnerf_data/scenes'
    args_edict.name_scene = 'scene0758_00'
    #args_edict.scene_sizes = [[-2.9, 2.9], [-3.1, 3.2], [-1.9, 1.4]]
    args_edict.scene_sizes = [[-2.1, 2.3], [-3.2, 2.4], [-1.8, 1.6]]
    args_edict.voxel_sizes = [0.016, 0.08, 0.36, 0.96]
    
    #args_edict.sdf_feature_dim = [8, 4, 4, 4]
    #args_edict.rgb_feature_dim = [12, 4, 4, 4]

    args_edict.sdf_feature_dim = [4, 4, 4, 4]
    args_edict.rgb_feature_dim = [8, 4, 4, 4]

    args_edict.near = 0.1
    args_edict.far = 8.
    
    args_edict.batch_size = 4096
    
    args_edict.seed = 1234
    
    args_edict.iterations = 10000
    args_edict.iter_print = 100
    args_edict.iter_viz = 500
    args_edict.iter_eval = 500
    args_edict.iter_save = 11000
        
    args_edict.lr = edict()
    args_edict.lr.decoder = 0.001
    args_edict.lr.features = 0.01
    args_edict.lr.inv_s = 0.001

    args_edict.rendering = edict()
    args_edict.rendering.H = 468
    args_edict.rendering.W = 624
    args_edict.rendering.flip_x = False
    args_edict.rendering.flip_y = False
    args_edict.rendering.inverse_y = False
    args_edict.crop = 0

    args_edict.geometry = edict()
    args_edict.geometry.W = 128
    args_edict.geometry.D = 1
    args_edict.geometry.skips = list()
    args_edict.geometry.n_freq = 4
    
    args_edict.radiance = edict()
    args_edict.radiance.W = 128
    args_edict.radiance.D = 1
    args_edict.radiance.skips = list()
    args_edict.radiance.n_freq = 4

    args_edict.color_weight = 1.
    args_edict.depth_weight = 0.12
    args_edict.normal_weight = 0.1
    args_edict.asdf_weight = 0.2
    args_edict.eikonal_weight = 0.15

    args_edict.use_eikonal_only = False
    
    args_edict.asdf_loss = edict()
    args_edict.asdf_loss.growing_trunc_init_val = 8
    args_edict.asdf_loss.growing_trunc_last_iter = 150
    args_edict.asdf_loss.growing_trunc_last_val = 0.09

    args_edict.n_samples = 128
    args_edict.n_importance = 64
    args_edict.n_importance_step_size = 16

    args_edict.inv_s = 0.3
    
    return args_edict  
