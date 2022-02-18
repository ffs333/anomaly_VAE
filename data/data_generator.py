import os
from random import shuffle

import torch
import torchaudio as ta


def spectrogram_generator(path_dir, step, chan, secs=3):
    data = []
    spec_size = int(256 * secs / 3)
    folder_items = os.listdir(path_dir)
    mfcc_transform = ta.transforms.MFCC(sample_rate=48000,
                                        n_mfcc=32, melkwargs={'n_fft': 2048,
                                                              'n_mels': 256,
                                                              'hop_length': 563})
    print(f'Settings: \n'
          f'Folder: {path_dir}\n'
          f'{len(os.listdir(path_dir))} Items in folder\n'
          f'Step: {step}\n'
          f'Actual channel: {chan}\n'
          f'Chunk length: {secs}\n')

    for i in range(len(folder_items)):
        if folder_items[i].split('.')[-1] != 'wav':
            print(f'Item #{i+1} {folder_items[i]} not a .wav file')
            continue
        rec, sr = ta.load(os.path.join(path_dir, folder_items[i]))
        if sr != 48000:
            print(f'{folder_items[i]} sample rate is {sr}, not 48000')
            continue
        print(f'Item #{i + 1} Record: {folder_items[i]}, Length: {rec.size(1)}, '
              f'Parts: {int((rec.size(1)/sr - secs) / step)}')
        rec = rec[chan].unsqueeze(0)
        for idx in range(0, int(rec.size(1) / sr), step):
            a = rec[:, idx * sr:(idx + secs) * sr]
            a = mfcc_transform(a)
            a = a[:, :, :spec_size]
            if a.size(2) == spec_size:
                data.append(a)
        print(f'Data current length: {len(data)}')
    print('Spectrogram generation finished')
    return data


def data_pipeline(config):
    full_data = spectrogram_generator(config['data_folder'], config['step'], config['channel'], config['sec_len'])
    if config['split_train_test']:
        shuffle(full_data)
        train_data = full_data[:int((1 - config['split_train_test'])*len(full_data))]
        eval_data = full_data[int((1 - config['split_train_test'])*len(full_data)):]
        print(f'Train data length: {len(train_data)}\n'
              f'Eval data length: {len(eval_data)}')
        torch.save(train_data, os.path.join(config['data_output_folder'], 'train_data.pt'))
        torch.save(eval_data, os.path.join(config['data_output_folder'], 'eval_data.pt'))
    else:
        print(f'Data length: {len(full_data)}')
        torch.save(full_data, os.path.join(config['data_output_folder'], 'train_data.pt'))
    print(f'Data saved here: {config["data_output_folder"]}')