import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


from data.dataset import BaseDataset
from core.modules.utils import make_collate_fn
from model.ae_model import AE
from model.ae_model_light import AELight
from model.classification_model import BaseCNN
from core.train import training, training_class
from core.modules.utils import  uniquify, config_dump


def prepare(config, raw_config):
    """
    Prepare training process
    :param config: prepared configuration data
    :param raw_config: raw config file
    """
    if config['type'] == 'vae':
        model = AE(config.input_size,
                   dropout_pb=config.dropout,
                   batch_size=config.batch_size)
    elif config['type'] == 'vae_light':
        model = AELight(config.input_size,
                        dropout_pb=config.dropout,
                        batch_size=config.batch_size)
    elif config['type'] == 'classification':
        model = BaseCNN(config.input_size,
                        dropout_pb=config.dropout,
                        num_classes=config.num_classes)
    else:
        raise KeyError('Type should be one of "vae", "vae_light", "classification"')

    if config.transfer:
        model.load_state_dict(torch.load(config.checkpoint))

    train_dataset = BaseDataset(config.train_data)
    eval_dataset = BaseDataset(config.eval_data)
    print(f'Train dataset size: {len(train_dataset)}\n'
          f'Eval dataset size: {len(eval_dataset)}')
    print(f'With BATCH SIZE: {config.batch_size}')
    if 'device' in config:
        if isinstance(config.device, int):
            device = f'cuda:{config.device}'
        else:
            device = f'{config.device}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    device = torch.device(device)

    model = model.to(device)

    train_dataloader = DataLoader(
                                train_dataset,
                                config.batch_size,
                                shuffle=True,
                                collate_fn=make_collate_fn(device, config['type'])
                                    )

    eval_dataloader = DataLoader(
                                eval_dataset,
                                config.batch_size,
                                shuffle=False,
                                collate_fn=make_collate_fn(device, config['type'])
                                    )
    opti = {
        'adam': Adam(model.parameters(), lr=config.learning_rate),
        'adamw': AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    }

    optimizer = opti[config.optimizer]

    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config.num_epochs,
                                  eta_min=config.learning_rate * 0.03,
                                  last_epoch=-1)

    path_dump, num_dump = uniquify(f'{config.output_config}/config_{config["type"]}.cfg')
    config_dump(path_dump, raw_config)
    print(f'Current train loop config saved here: {path_dump}')

    if config['type'] in ['vae', 'vae_light']:
        training(model, config.num_epochs, train_dataloader, eval_dataloader, optimizer, scheduler,
                 config.output_check, config.output_spec, config.eval_step_epochs, config.mfcc_step_epochs,
                 config.mel_step_epochs, config.save_epoch,
                 [path_dump, num_dump, raw_config, config.output_config, config["type"]])
    elif config['type'] == 'classification':
        training_class(model, config.num_epochs, train_dataloader, eval_dataloader, optimizer, scheduler,
                       config.output_check, config.eval_step_epochs, config.save_epoch,
                       [path_dump, num_dump, raw_config, config.output_config, config["type"]], config.num_classes)