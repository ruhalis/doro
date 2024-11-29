import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_filters=32):
        super(Generator, self).__init__()
        
        # Initial convolution block
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 7, padding=3),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters*4),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*8, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters*8),
            nn.ReLU(inplace=True)
        )
        
        # Add attention mechanism
        self.attention = SelfAttentionBlock(base_filters*8)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_filters*8, dropout_rate=0.2),
            ResidualBlock(base_filters*8, dropout_rate=0.2),
            self.attention,  # Add attention after some residual blocks
            ResidualBlock(base_filters*8, dropout_rate=0.2),
            ResidualBlock(base_filters*8, dropout_rate=0.2)
        )
        
        # Upsampling with correct channel sizes
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*8, base_filters*4, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_filters*4),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*4, base_filters*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*2, base_filters, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.output = nn.Sequential(
            nn.Conv2d(base_filters, output_channels, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoding
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        # Residual blocks
        x = self.res_blocks(x)
        
        # Decoding (in correct order)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        
        # Add high-frequency enhancement
        output = self.output(x)
        enhanced = self.enhance_high_frequency(output)
        return enhanced
    
    def enhance_high_frequency(self, x):
        # Apply unsharp masking
        blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - blur
        sharpened = x + 0.5 * high_freq  # Adjust factor (0.5) to control sharpening intensity
        return sharpened

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x) 

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        q = self.query(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, width*height)
        v = self.value(x).view(batch_size, -1, width*height)
        
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=2)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        return self.gamma * out + x 