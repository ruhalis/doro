import torch
import torch.nn as nn
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        blocks = []
        blocks.extend([i for i in vgg.features[:4].children()])
        blocks.extend([i for i in vgg.features[4:9].children()])
        blocks.extend([i for i in vgg.features[9:16].children()])
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = True

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
            
        return loss 