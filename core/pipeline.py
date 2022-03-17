from pyhocon import ConfigFactory

from core.train_pipeline import prepare


def prepare_config(path):
    """
    Read config file
    :param path: path to config file
    :return tuple with configurations for necessary modules
    """
    conf = ConfigFactory.parse_file(path)
    return conf['data'], conf['model'], conf['train'], conf


def go_pipeline(data_conf, model_conf, train_conf, config_raw):
    """
    Aggregate configs
    :param data_conf: data config
    :param model_conf: model params config
    :param train_conf: train params config
    :param config_raw: raw config file
    """
    work_config = train_conf
    work_config.put('train_data', data_conf.get_string('train_data'))
    work_config.put('eval_data', data_conf.get_string('eval_data'))
    work_config.put('batch_size', model_conf.get_int('batch_size'))
    work_config.put('input_size', model_conf.get_list('input_size'))
    work_config.put('dropout', model_conf.get_float('dropout_rate'))
    work_config.put('type', model_conf.get_string('model_type'))

    prepare(work_config, config_raw)