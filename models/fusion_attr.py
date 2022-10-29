import torch.nn as nn
import torch
from torch.nn import functional as F
from models.vit_lncnn import Encoder

class FusionModel(nn.Module):
    def __init__(self, ratio=0.6):
        super(FusionModel, self).__init__()
        c1 = int(64)
        c2 = int(128)
        c3 = int(256)
        c4 = int(512)

        self.block1 = Block([c1, c1, 'M'], in_channels=3, cross_channels = c1,vit_img_size_ = None)
        self.block2 = Block([c2, c2, 'M'], in_channels=c1,cross_channels = c2,vit_img_size_ = None)
        self.block3 = Block([c3, c3, c3, c3, 'M'], in_channels=c2,cross_channels = c3,vit_img_size_ = [32,2])
        self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3,cross_channels = c4,vit_img_size_ = [16,1])
        self.block5 = Block([c4, c4, c4, c4], in_channels=c4,cross_channels = c4,vit_img_size_ = [8,1])
        
        self.conv1d = nn.Sequential(
                     nn.Conv2d(in_channels=c4*2, out_channels=c4, kernel_size=(1,1), stride=(1, 1), bias=False)
                     )

        self.reg_layer = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self._initialize_weights()

    def forward(self, RGBT):
        RGB = RGBT[0]
        T = RGBT[1]
        #print ('input0-------------0000',RGB.shape,T.shape)  #[1, 3, 256, 256]

        RGB, T = self.block1(RGB, T)
        #print ('Block1------------11111',RGB.shape)         #[1, 64, 128, 128]
        RGB, T = self.block2(RGB, T)
        #print ('Block2------------22222',RGB.shape)         #[1, 128, 64, 64]
        RGB, T = self.block3(RGB, T)
        #print ('Block3------------33333',RGB.shape)         #[1, 256, 32, 32]
        RGB, T = self.block4(RGB, T)
        #print ('Block4------------44444',RGB.shape)         #[1, 512, 16, 16]
        RGB, T = self.block5(RGB, T)
        #print ('Block5------------55555',RGB.shape)         #[1, 512, 16, 16]
        
        x = torch.cat((RGB,T),dim=1)
        #print ('Block6------------66666',x.shape)
        x = self.conv1d(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, std=0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0) #module.bias.data.zero_() #
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


class Block(nn.Module):
    def __init__(self, cfg, in_channels, cross_channels,vit_img_size_):
        super(Block, self).__init__()
        

        self.rgb_conv = make_layers(cfg, in_channels=in_channels)
        self.t_conv = make_layers(cfg, in_channels=in_channels)
        
        
        self.Flag = False
        
        if len(cfg) > 3:
            self.Encoder = Encoder(dim=cross_channels,num_heads=4,num_x_layers = 1,vit_patch_size=vit_img_size_)
            self.Flag = True
       

    def forward(self, RGB, T):
        RGB = self.rgb_conv(RGB)
        T = self.t_conv(T)
        
        if (self.Flag):
        
            RGB_att,T_att = self.Encoder(RGB,T)
            RGB = RGB + RGB_att
            T = T + T_att
            
            
            #RGB,T = self.Encoder(RGB,T)
        
        
        
        return RGB, T


def fusion_model():
    model = FusionModel()
    return model


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
