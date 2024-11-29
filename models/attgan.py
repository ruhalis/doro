import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_filters=64):
        super(Generator, self).__init__()
        
        # Initial convolution block with more precise feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 7, padding=3),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1)
        )
        
        # Downsampling with skip connections
        self.down1 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, 3, padding=1),  # Additional conv
            nn.InstanceNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*4, base_filters*4, 3, padding=1),  # Additional conv
            nn.InstanceNorm2d(base_filters*4),
            nn.ReLU(inplace=True)
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*8, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*8, base_filters*8, 3, padding=1),  # Additional conv
            nn.InstanceNorm2d(base_filters*8),
            nn.ReLU(inplace=True)
        )
        
        # Enhanced wrinkle attention mechanism
        self.wrinkle_attention = nn.Sequential(
            nn.Conv2d(base_filters*8, base_filters*8, 1),
            nn.InstanceNorm2d(base_filters*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*8, base_filters*8, 1),
            nn.Sigmoid()
        )
        
        # Multi-scale feature refinement
        self.refine_attention = SelfAttentionBlock(base_filters*8)
        
        # More residual blocks for better feature processing
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_filters*8, dropout_rate=0.1),  # Reduced dropout
            ResidualBlock(base_filters*8, dropout_rate=0.1),
            ResidualBlock(base_filters*8, dropout_rate=0.1),
            ResidualBlock(base_filters*8, dropout_rate=0.1),
            ResidualBlock(base_filters*8, dropout_rate=0.1),
            ResidualBlock(base_filters*8, dropout_rate=0.1)
        )
        
        # Precise upsampling path
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*8, base_filters*4, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*4, base_filters*4, 3, padding=1),  # Additional conv
            nn.InstanceNorm2d(base_filters*4),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*4, base_filters*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, 3, padding=1),  # Additional conv
            nn.InstanceNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*2, base_filters, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),  # Additional conv
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Refined output for precise skin texture
        self.output = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, output_channels, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x, intensity=1.0):
        B, C, H, W = x.shape
        
        # Process in chunks if image is too large
        if H > 512 or W > 512:
            return self.forward_chunks(x, intensity)
            
        # Normal forward pass for smaller images
        x0 = self.initial(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        wrinkle_mask = self.wrinkle_attention(x3)
        x3 = x3 * (1 + intensity * wrinkle_mask)
        
        x3 = self.refine_attention(x3)
        x3 = self.res_blocks(x3)
        
        x = self.up3(x3) + x2
        x = self.up2(x) + x1
        x = self.up1(x) + x0
        
        return self.output(x)

    def forward_chunks(self, x, intensity=1.0):
        """Process large images in chunks"""
        B, C, H, W = x.shape
        chunk_size = 512  # Process 512x512 chunks
        output = torch.zeros_like(x)
        
        for h in range(0, H, chunk_size//2):  # 50% overlap
            for w in range(0, W, chunk_size//2):
                h_start = h
                w_start = w
                h_end = min(h + chunk_size, H)
                w_end = min(w + chunk_size, W)
                
                # Extract chunk with padding
                pad_h = max(0, chunk_size - (h_end - h_start))
                pad_w = max(0, chunk_size - (w_end - w_start))
                chunk = x[:, :, h_start:h_end, w_start:w_end]
                
                if pad_h > 0 or pad_w > 0:
                    chunk = F.pad(chunk, (0, pad_w, 0, pad_h), mode='reflect')
                
                # Process chunk
                with torch.cuda.amp.autocast():
                    processed_chunk = super().forward(chunk, intensity)
                
                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    processed_chunk = processed_chunk[:, :, :h_end-h_start, :w_end-w_start]
                
                # Blend chunks with overlap
                if h > 0:  # Blend vertical overlap
                    blend_h = chunk_size//4
                    alpha = torch.linspace(0, 1, blend_h).view(1, 1, blend_h, 1).to(x.device)
                    output[:, :, h:h+blend_h, w_start:w_end] = (
                        output[:, :, h:h+blend_h, w_start:w_end] * (1 - alpha) +
                        processed_chunk[:, :, :blend_h, :] * alpha
                    )
                    processed_chunk = processed_chunk[:, :, blend_h:, :]
                    h_start += blend_h
                
                if w > 0:  # Blend horizontal overlap
                    blend_w = chunk_size//4
                    alpha = torch.linspace(0, 1, blend_w).view(1, 1, 1, blend_w).to(x.device)
                    output[:, :, h_start:h_end, w:w+blend_w] = (
                        output[:, :, h_start:h_end, w:w+blend_w] * (1 - alpha) +
                        processed_chunk[:, :, :, :blend_w] * alpha
                    )
                    processed_chunk = processed_chunk[:, :, :, blend_w:]
                    w_start += blend_w
                
                # Copy remaining chunk
                output[:, :, h_start:h_end, w_start:w_end] = processed_chunk
                
                # Clear cache
                torch.cuda.empty_cache()
        
        return output

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