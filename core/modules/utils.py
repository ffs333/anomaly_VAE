#!/usr/bin/env
import os
import json

import torch

from typing import Callable, Sequence


def make_sequential_collate_fn(device):
    """
    Factory function to create collate function for sequential models
    :param device: PyTorch device entity. This is context for CPU/GPU software environment
    :param seq_elem_size: size of sequential element
    :return collate function
    """

    def collate_fn(batch):
        """
        collate function is necessary for processing variable length tensors.
        It pads missed parts and return padding mask (1 for real data and 0 for padded item).
        Also this function passes all tensors to device
        :param batch: Input tensor with labels and data samples
        :return tuple with labels and batch tensors, padding mask tensor and padding lengths for sequence models
        """

        # repack labels and batch into separate entities
        if len(batch[0]) == 2:
            meta = [(5 * e[0]).type(torch.float32).unsqueeze(0) for e in batch]

            proc_meta = torch.cat(meta, dim=0).to(device)
        batch = [e['data'] for e in batch]
        proc_batch = torch.cat(batch, dim=0).to(device)

        proc_batch = proc_batch.unsqueeze(1)

        if 'proc_meta' in locals():
            return proc_batch, proc_meta
        else:
            return proc_batch

    return collate_fn


def make_classification_collate_fn(device):
    """
    Factory function to create collate function for sequential models
    :param device: PyTorch device entity. This is context for CPU/GPU software environment
    :return collate function
    """

    def collate_fn(batch):
        """
        collate function is necessary for processing variable length tensors.
        It pads missed parts and return padding mask (1 for real data and 0 for padded item).
        Also this function passes all tensors to device
        :param batch: Input tensor with labels and data samples
        :return tuple with labels and batch tensors, padding mask tensor and padding lengths for sequence models
        """

        # repack labels and batch into separate entities
        labels = torch.tensor([e['label'] for e in batch], dtype=torch.float32).to(device)
        proc_batch = [e['data'] for e in batch]
        proc_batch = torch.cat(proc_batch, dim=0).to(device)

        proc_batch = proc_batch.unsqueeze(1)

        return proc_batch, labels

    return collate_fn


def make_collate_fn(device, type_of):
    """
    Factory function for creating collate function based on model
    :param type_of:
    :param device: torch device entity to transfer data on
    :return collate function
    """
    if type_of == 'classification':
        return make_classification_collate_fn(device)
    else:
        return make_sequential_collate_fn(device)


def process_rec(idx, recs: Sequence, preproc_fn: Callable = None):
    """
    Preprocess loaded record and converts it to torch tensor
    :param idx: index of preprocessed record
    :param recs: sequence of dicts with sample info
    :param preproc_fn: preprocessing function (optional)
    :return torch tensor with data
    """
    return recs[idx]


def uniquify(path):
    """
    Make unique path if file exists
    :param path: path to save file
    :return new path available and added number
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path, counter-1


def config_dump(path, conf_file):
    """
    Dump config file
    :param path: path to save file
    :param conf_file: config data
    """
    if not os.path.exists((path.split('/config_exp')[0])):
        os.makedirs(path.split('/config_exp')[0])
    with open(path, 'w') as f:
        json.dump(conf_file, f)
