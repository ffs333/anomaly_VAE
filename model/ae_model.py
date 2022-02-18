import torch
import torch.nn as nn

from tools.types_ import *
from model.base import BaseAE


class AE(BaseAE):

    def __init__(self, input_shape,
                 dropout_pb=0.1,
                 hidden_dims: List = None,
                 batch_size=30
                 ):

        super(AE, self).__init__()
        self._shape = input_shape
        self._batch_size = batch_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        self._latent_dims = hidden_dims[-1] * 2

        # Encoder
        self._in_block = nn.Sequential(
            nn.Conv2d(1, hidden_dims[0], kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_pb),
            nn.MaxPool2d(kernel_size=2)

        )

        modules = []
        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, padding_mode='circular'),
                    nn.BatchNorm2d(dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_pb),
                    nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, padding_mode='circular'),
                    nn.BatchNorm2d(dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_pb),
                    nn.MaxPool2d(kernel_size=2)
                )
            )
        self._encoder = nn.Sequential(*modules)

        h, w = input_shape
        dim_scale_factor = 2 ** (len(modules) + 1)
        self.out_dims = torch.tensor((h // dim_scale_factor, w // dim_scale_factor, hidden_dims[-1]*2))
        encoder_out_size = torch.prod(self.out_dims).item()

        self._enc_out = nn.Sequential(nn.Linear(encoder_out_size, hidden_dims[-1] * 16),
                                      nn.ReLU(),
                                      nn.Dropout(dropout_pb))

        self._fc_mu = nn.Linear(hidden_dims[-1] * 16, 128)
        self._fc_var = nn.Linear(hidden_dims[-1] * 16, 128)

        # Decoder

        self._decoder_input = nn.Sequential(
                                    nn.Linear(128, hidden_dims[-1] * 16),
                                    nn.ReLU(),
                                    nn.Dropout(dropout_pb),
                                    nn.Linear(hidden_dims[-1] * 16, encoder_out_size)
                                        )

        hidden_dims.reverse()
        modules = []
        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dim * 2, dim, kernel_size=3, padding=1, output_padding=0,
                                       stride=1, padding_mode='zeros'),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_pb),
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='circular'),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_pb),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
            )

        self._decoder = nn.Sequential(*modules)

        self._final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_pb),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_pb),
            nn.Conv2d(hidden_dims[-1], 1, kernel_size=3, padding=1, padding_mode='circular')
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        :param input: (Tensor) input tensor to encoder [N x C x H x W]
        :param meta: (Tensor) input tensor of metadata [N x W]
        :return: (Tensor) list of latent codes
        """

        result = self._in_block(input)
        result = self._encoder(result)
        result = torch.flatten(result, start_dim=1)

        # Concatenate meta data with encoder output
        # result = torch.cat((result, meta), dim=1)
        result = self._enc_out(result)
        mu = self._fc_mu(result)
        log_var = self._fc_var(result)

        return [mu, log_var]

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor) -> Tensor:
        """
        :param z: (Tensor) input tensor to decoder [B x D]
        :return: (Tensor) output tensor from decoder [B x C x H x W]
        """
        result = self._decoder_input(z)

        batch_size = result.size()[0]
        result = result.view(batch_size, self.out_dims[2], self.out_dims[0], self.out_dims[1])

        result = self._decoder(result)

        return self._final_layer(result)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        :param input: (Tensor) audio spectrogram [N x C x H x W]
        :param meta: (Tensor) meta data [N x W]
        :return: model output: (list) list of tensors of input and output audio spectrogram [N x C x H x W]
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        return [self.decode(z), input, mu, log_var]

    @staticmethod
    def loss_function(*args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args: prediction, input, mu, logvar
        :param kwargs: batch size
        :return: weighted loss sum, reconstruction loss, KLD loss
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = torch.nn.functional.mse_loss(recons, input)
        if len(mu.size()) == 1:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=0), dim=0)
        else:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss * 100
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def batch_size(self) -> int:
        return self._batch_size
