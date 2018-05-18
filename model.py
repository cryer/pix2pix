import torch.nn as nn
import torch.nn.functional as F
import torch
from unet_block import *

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# build U-Net using blocks
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.downsample1 = UNetDown(in_channels, 64, normalize=False)
        self.downsample2 = UNetDown(64, 128)
        self.downsample3 = UNetDown(128, 256)
        self.downsample4 = UNetDown(256, 512)
        self.downsample5 = UNetDown(512, 512)
        self.downsample6 = UNetDown(512, 512)
        self.downsample7 = UNetDown(512, 512)
        self.downsample8 = UNetDown(512, 512, normalize=False)

        self.upsample1 = UNetUp(512, 512, dropout=0.5)
        self.upsample2 = UNetUp(1024, 512, dropout=0.5)
        self.upsample3 = UNetUp(1024, 512, dropout=0.5)
        self.upsample4 = UNetUp(1024, 512, dropout=0.5)
        self.upsample5 = UNetUp(1024, 256)
        self.upsample6 = UNetUp(512, 128)
        self.upsample7 = UNetUp(256, 64)


        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )


    def forward(self, x):
        down1 = self.downsample1(x)
        down2 = self.downsample2(down1)
        down3 = self.downsample3(down2)
        down4 = self.downsample4(down3)
        down5 = self.downsample5(down4)
        down6 = self.downsample6(down5)
        down7 = self.downsample7(down6)
        down8 = self.downsample8(down7)
        up1 = self.upsample1(down8, down7)
        up2 = self.upsample2(up1, down6)
        up3 = self.upsample3(up2, down5)
        up4 = self.upsample4(up3, down4)
        up5 = self.upsample5(up4, down3)
        up6 = self.upsample6(up5, down2)
        up7 = self.upsample7(up6, down1)

        return self.final(up7)


#Discriminator

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)