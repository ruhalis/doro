import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block without Dropout
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

# Self-Attention Block (Adjusted)
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        self.in_channels = in_channels

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Query, Key, Value
        proj_query = self.query_conv(x).view(batch_size, -1, H * W)  # B x C x N
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)      # B x C x N
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)  # B x C x N

        # Compute attention
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)    # B x N x N
        attention = self.softmax(energy)                             # B x N x N

        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))      # B x C x N
        out = out.view(batch_size, C, H, W)

        out = self.gamma * out + x
        return out

# Generator
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_filters=64):
        super(Generator, self).__init__()

        # Initial convolution block
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_filters, 7),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_filters * 4) for _ in range(6)]
        )

        # Self-Attention
        self.attention = SelfAttentionBlock(base_filters * 4)

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, output_channels, 7),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.res_blocks(x3)
        x4 = self.attention(x4)

        x5 = self.up1(x4)
        x6 = self.up2(x5)

        out = self.output(x6)
        return out

# Discriminator with Spectral Normalization
class Discriminator(nn.Module):
    def __init__(self, input_channels=3, base_filters=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, base_filters, normalize=False),
            *discriminator_block(base_filters, base_filters * 2),
            *discriminator_block(base_filters * 2, base_filters * 4),
            *discriminator_block(base_filters * 4, base_filters * 8),
            nn.Conv2d(base_filters * 8, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

