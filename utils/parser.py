import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_log', type=str, default='base')
    parser.add_argument('--dataset', type=str, default='mri')
    parser.add_argument('--data_path', type=str, default='/')
    parser.add_argument('--name_scene', type=str, default='sub001')
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)

    return parser.parse_args()
