import os
import importlib

def load_config(args):

    config_path = f'config.{args.dataset}.{args.name_scene}'
    
    config_module = importlib.import_module(config_path)
    new_args = config_module.get_args()

    # update easydict with input arguments
    for key in vars(args).keys():
        new_args[key] = vars(args)[key]
        
    return new_args
