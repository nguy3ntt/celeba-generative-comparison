import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for 64x64 RGB images.

    Input:
        image tensor of shape (batch, 3, 64, 64)

    Output:
        reconstructed image tensor of shape (batch, 3, 64, 64)
        latent mean
        latent log variance
    """

    def __init__(
        self,
        image_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # encoder
        encoder_layers = []
        in_channels = image_channels

        for h_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # After four stride-2 conv layers:
        # 64 -> 32 -> 16 -> 8 -> 4
        self.flatten_dim = hidden_dims[-1] * 4 * 4

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        hidden_dims_reverse = hidden_dims[::-1]

        decoder_layers = []

        for i in range(len(hidden_dims_reverse) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims_reverse[i],
                        hidden_dims_reverse[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims_reverse[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )

        self.decoder = nn.Sequential(*decoder_layers)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims_reverse[-1],
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def encode(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)

        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std

        return z

    def decode(self, z):
        decoded = self.decoder_input(z)
        decoded = decoded.view(-1, self.hidden_dims[-1], 4, 4)
        decoded = self.decoder(decoded)
        reconstructed = self.final_layer(decoded)

        return reconstructed

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)

        return reconstructed, mu, logvar

    def sample(self, num_samples: int, device: torch.device):
        """
        Generate new samples from random latent vectors.
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)

        return samples