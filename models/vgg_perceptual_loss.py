import torch
import torch.nn as nn
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        self.resize = resize

        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        # Assuming you want to use relu1_2, relu2_2, relu3_3, relu4_3 layers
        for x in range(0, 4):   # relu1_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):   # relu2_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):  # relu3_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23): # relu4_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # Freeze the weights
        for param in self.parameters():
            param.requires_grad = False

        # Register mean and std as buffers
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, input, target):
        # Ensure inputs are 3-channel images
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] != 3:
            target = target.repeat(1, 3, 1, 1)

        # Normalize inputs
        input = (input - self.mean.to(input.device)) / self.std.to(input.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)

        # Resize if necessary
        if self.resize:
            input = nn.functional.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = nn.functional.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        # Compute features
        input_features = []
        target_features = []
        x = input
        y = target

        x = self.slice1(x)
        y = self.slice1(y)
        input_features.append(x)
        target_features.append(y)

        x = self.slice2(x)
        y = self.slice2(y)
        input_features.append(x)
        target_features.append(y)

        x = self.slice3(x)
        y = self.slice3(y)
        input_features.append(x)
        target_features.append(y)

        x = self.slice4(x)
        y = self.slice4(y)
        input_features.append(x)
        target_features.append(y)

        # Compute perceptual loss as sum of L1 losses between features
        loss = 0.0
        for input_feat, target_feat in zip(input_features, target_features):
            loss += nn.functional.l1_loss(input_feat, target_feat)

        return loss
