import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as f
from src.dataset import RetinaVesselDataset



def doubleConvolutional(in_channels,out_channels):
    double_conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1),
        nn.ReLU(inplace=True)
    )
    return double_conv

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # downstream
        self.dc1 = doubleConvolutional(3,64)
        self.dc2 = doubleConvolutional(64,128)
        self.dc3 = doubleConvolutional(128,256)
        self.dc4 = doubleConvolutional(256,512)
        self.dc5 = doubleConvolutional(512,1024)

        # Expanding path
        self.up1 = nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
        self.updc1 = doubleConvolutional(1024,512)
        self.up2 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
        self.updc2 = doubleConvolutional(512,256)
        self.up3 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.updc3 = doubleConvolutional(256,128)
        self.up4 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        self.updc4 = doubleConvolutional(128,64)
        
        # output
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1
        )

    def forward(self,x):
        down_1 = self.dc1(x)
        downpool1 = self.maxpool(down_1)
        down_2 = self.dc2(downpool1)    
        downpool2 = self.maxpool(down_2)
        down_3 = self.dc3(downpool2)
        downpool3 = self.maxpool(down_3)
        down_4 = self.dc4(downpool3)
        downpool4 = self.maxpool(down_4)
        down_5 = self.dc5(downpool4)

        up_1 = self.up1(down_5)
        x = self.updc1(torch.cat([down_4, up_1], dim=1))
        up_2 = self.up2(x)
        x = self.updc2(torch.cat([down_3, up_2], dim=1))
        up_3 = self.up3(x)
        x = self.updc3(torch.cat([down_2, up_3], dim=1))
        up_4 = self.up4(x)
        x = self.updc4(torch.cat([down_1, up_4], dim=1))

        out = self.out(x)
        return out

if __name__ == '__main__':
    
    ds = RetinaVesselDataset(None, "train", "splits/train.txt")
    train_image,mask_image = ds[3]
    model = Unet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total trainable parameters: {total_trainable_params}")
    outputs = model(train_image)
    print(outputs.shape)
