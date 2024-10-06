import os
import logging

from datetime import datetime

def set_logger(args):
    
    args.log_dir = os.path.join('logs',
                                args.name_scene,
                                args.name_log)

    # create dirs
    os.makedirs('logs', exist_ok=True)
    os.makedirs(os.path.join('logs', args.name_scene), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
         
    logger = logging.getLogger()       
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    now = datetime.now()
    time_now = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    file_handler = logging.FileHandler(os.path.join(args.log_dir, f'{time_now}_mylog.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
