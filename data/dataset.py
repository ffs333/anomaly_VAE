#!/usr/bin/env
from typing import Callable
import torch
from torch.utils.data import Dataset

from core.modules.utils import process_rec


class BaseDataset(Dataset):
    """
    This dataset subclass is used for reading data records
    and passing it to corresponding models in specified shape
    """
    def __init__(self, path, preproc_fn: Callable = None):
        """
        Constructor
        :param path: path to dataset file (.npy)
        :param preproc_fn: preprocessing function. Should return new data and new shape. (optional)
        """
        self._data = torch.load(path)
        self._preproc_fn = preproc_fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return process_rec(idx, self._data, self._preproc_fn)