import os
import argparse
import torch
import random
import logging
import math

from trainer.trainer import run_trainer

from utils.parser import arg_parser
from utils.logger import set_logger

from config.utils import load_config

def main(args, logger):

    for key in args.keys():
        logger.info(f'{key}: {args[key]}')

    run_trainer(args, logger)

if __name__ == '__main__':
    args = arg_parser()
    new_args = load_config(args)
    
    logger = set_logger(new_args)
    main(new_args, logger)
