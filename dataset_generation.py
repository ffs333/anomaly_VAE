import argparse

from core.pipeline import prepare_config
from data.data_generator import data_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to run training process')

    parser.add_argument('-c', '--config', type=str,
                        required=True,
                        help='path to config file')

    return parser.parse_args()


if __name__ == '__main__':
    print('Start data generation process.')
    args = parse_args()
    data_conf, _, _ = prepare_config(args.config)
    data_pipeline(data_conf)
