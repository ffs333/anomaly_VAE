#!/usr/bin/env
import torch
from metrics.metric import Metric


class AccuracyMetric(Metric):
    """
    Implementation of simple accuracy metric
    """
    def __init__(self):
        super(AccuracyMetric, self).__init__()
        self._total_elements = 0
        self._total_td = None

    def update(self, pred, target):
        if self._total_td is None:
            self._total_td = (target == pred).sum()
        else:
            self._total_td += (target == pred).sum()
        self._total_elements += len(pred) if pred.dim() > 0 else 1

    def flush(self):
        self._total_td, self._total_elements = None, 0

    def compute(self):
        print(f'Accuracy: {torch.true_divide(self._total_td, self._total_elements)}')
        return torch.true_divide(self._total_td, self._total_elements)