import mlx.core as mx
import mlx.nn as nn


class Critic(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Conv2d(in_channels=512, out_channels=output_dim, kernel_size=4, stride=1, padding=0),
            lambda x: x.reshape(x.shape[0], -1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


class Generator(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):

        super().__init__()

        self.net = nn.Sequential(
            lambda x: x.reshape(-1, 1, 1, input_dim),
            nn.ConvTranspose2d(in_channels=input_dim, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm(num_features=512, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm(num_features=256, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm(num_features=128, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm(num_features=64, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=64, out_channels=output_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


