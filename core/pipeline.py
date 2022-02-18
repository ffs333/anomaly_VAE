from pyhocon import ConfigFactory

from core.train_pipeline import prepare


def prepare_config(path):
    """
    Read config file
    :param path: path to config file
    :return tuple with configurations for necessary modules
    """
    conf = ConfigFactory.parse_file(path)
    return conf['data'], conf['model'], conf['train']


def go_pipeline(data_conf, model_conf, train_conf):
    """
    Aggregate configs
    :param data_conf: data config
    :param model_conf: model params config
    :param train_conf: train params config
    """
    config = train_conf
    config.put('train_data', data_conf.get_string('train_data'))
    config.put('eval_data', data_conf.get_string('eval_data'))
    config.put('batch_size', model_conf.get_int('batch_size'))
    config.put('input_size', model_conf.get_list('input_size'))
    config.put('dropout', model_conf.get_float('dropout_rate'))
    config.put('type', model_conf.get_string('model_type'))

    prepare(config)
