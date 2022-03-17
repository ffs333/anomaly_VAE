import argparse

from core.pipeline import prepare_config
from core.pipeline import go_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to run training process')

    parser.add_argument('-c', '--config', type=str,
                        required=True,
                        help='path to config file')

    return parser.parse_args()


if __name__ == '__main__':
    print('Start training process.', 'Info prints via TensorBoard', sep='\n')
    args = parse_args()
    data_conf, model_conf, train_conf, conf = prepare_config(args.config)
    go_pipeline(data_conf, model_conf, train_conf, conf)
