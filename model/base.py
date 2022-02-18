from abc import abstractmethod
import torch.nn as nn

from tools.types_ import *


class BaseAE(nn.Module):

    def __init__(self) -> None:
        super(BaseAE, self).__init__()

    def encode(self, input: Tensor, meta: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass