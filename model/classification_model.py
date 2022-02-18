import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """
    Basic block for cnn model
    """
    def __init__(self, in_ch, out_ch, h, w, dropout_pb=.0):
        super(CNNBlock, self).__init__()

        self._block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_pb),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_pb),
            nn.MaxPool2d(kernel_size=2)
        ).apply(weights_init)

    def forward(self, batch):
        return self._block(batch)


class BaseCNN(nn.Module):

    def __init__(self, shape, dropout_pb=.0, num_classes=1):
        """
        Constructor
        :param shape: tuple with input shape
        :param dropout_pb: dropout rate
        """
        super(BaseCNN, self).__init__()
        h, w = shape
        self._num_classes = num_classes

        # first block
        channels_num = 32
        self._in_block = nn.Sequential(
            nn.Conv2d(1, channels_num, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Dropout(dropout_pb),
            nn.BatchNorm2d(channels_num),
            nn.MaxPool2d(kernel_size=2)
        )

        modules = []
        count = 1
        while channels_num < 512:
            modules.append(CNNBlock(channels_num,
                                    2 * channels_num,
                                    int(h * (0.5 ** count)),
                                    int(w * (0.5 ** count)),
                                    dropout_pb * (0.5 ** count)))
            channels_num *= 2
            count += 1

        self._main_block = nn.Sequential(*modules)
        dim_scale_factor = 2 ** (len(modules) + 1)
        self._out_dims = (h // dim_scale_factor, w // dim_scale_factor, channels_num)
        self._logits = nn.Sequential(
                                    nn.Linear(self._out_dims[0]*self._out_dims[1]*self._out_dims[2], 256),
                                    nn.ReLU(),
                                    nn.Dropout(dropout_pb),
                                    nn.Linear(256, num_classes))
        if num_classes == 1:
            self._loss = nn.BCELoss()
        else:
            self._loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        batch = self._in_block(batch)
        batch = self._main_block(batch)
        batch = torch.flatten(batch, start_dim=1)
        if self._num_classes == 1:
            return torch.sigmoid(self._logits(batch))
        return self._logits(batch)

    @property
    def out_dims(self):
        return self._out_dims

    def loss_function(self, inp, target):
        return self._loss(inp, target)


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)