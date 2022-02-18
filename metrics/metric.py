#!/usr/bin/env
from abc import abstractmethod


class Metric:
    """
    Base class for metrics in this framework
    """
    def __init__(self):
        pass

    @abstractmethod
    def update(self, preds, targets):
        """
        Updates metric for every batch.
        :param preds: tensor with predicted values
        :param targets: tensor with ground truth labels
        """
        raise NotImplemented

    @abstractmethod
    def compute(self):
        """
        Compute metric for current state
        :return metric value
        """
        raise NotImplemented

    @abstractmethod
    def flush(self):
        """
        Flush all metrics values to reuse instance of this class.
        It is useful in training stage.
        """
        raise NotImplemented
