import torch
import torch.nn as nn


def weights_init_normal(module):
    """
    DCGAN weight initialization

    The original DCGAN setup commonly initializes convolution and batch norm
    weights from a normal distribution
    """
    classname = module.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator for 64x64 RGB face generation

    Input:
        noise tensor of shape (batch, noise_dim, 1, 1)

    Output:
        generated image tensor of shape (batch, 3, 64, 64)
    """

    def __init__(
        self,
        noise_dim: int = 100,
        image_channels: int = 3,
        features_g: int = 64,
    ):
        super().__init__()

        self.noise_dim = noise_dim

        self.net = nn.Sequential(
            # input: (N, noise_dim, 1, 1)
            nn.ConvTranspose2d(noise_dim, features_g * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            # (N, features_g*8, 4, 4)

            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            # (N, features_g*4, 8, 8)

            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            # (N, features_g*2, 16, 16)

            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            # (N, features_g, 32, 32)

            nn.ConvTranspose2d(features_g, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # (N, 3, 64, 64)
        )

    def forward(self, noise):
        return self.net(noise)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator for 64x64 RGB images

    Input:
        image tensor of shape (batch, 3, 64, 64)

    Output:
        logits of shape (batch,)
    """

    def __init__(
        self,
        image_channels: int = 3,
        features_d: int = 64,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # Input: (N, 3, 64, 64)
            nn.Conv2d(image_channels, features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, features_d, 32, 32)

            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, features_d*2, 16, 16)

            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, features_d*4, 8, 8)

            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, features_d*8, 4, 4)

            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # (N, 1, 1, 1)
        )

    def forward(self, image):
        logits = self.net(image)
        return logits.view(-1)