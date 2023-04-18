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
        
        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_1.data.fill_(1.0)
        self.fuse_weight_2.data.fill_(1.0)
        self.conv2d_x = nn.Sequential(        
                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1, 1), padding=(1, 1), bias=False),
                     nn.ReLU(True),
                     nn.Dropout(0.5),
                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1, 1), padding=(1, 1), bias=False),
                     nn.ReLU(True),
                     nn.Dropout(0.5)       
                   )
        
        self.conv1d = nn.Sequential(
                     nn.Conv2d(in_channels=c4*2, out_channels=c4, kernel_size=(1,1), stride=(1, 1), bias=False)
                     )

        self.drop = nn.Dropout(0.5)  #self.drop = nn.Dropout(0.5)

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
        #with torch.no_grad():
        RGB, T = self.block1(RGB, T)
        #print ('Block1------------11111',RGB.shape)         #[1, 64, 128, 128]
        RGB, T = self.block2(RGB, T)
        #print ('Block2------------22222',RGB.shape)         #[1, 128, 64, 64]
        RGB, T,x_RGB_prompt1,x_T_prompt1 = self.block3(RGB, T)
        x_RGB_1,x_T_1 = RGB, T
        #print ('Block3------------33333',RGB.shape)         #[1, 256, 32, 32]
        RGB, T,x_RGB_prompt2,x_T_prompt2 = self.block4([RGB,x_RGB_prompt1], [T,x_T_prompt1])
        x_RGB_2,x_T_2 = RGB, T
        #print ('Block4------------44444',RGB.shape)         #[1, 512, 16, 16]
        RGB, T,x_RGB_prompt3,x_T_prompt3 = self.block5([RGB,x_RGB_prompt2], [T,x_T_prompt2])
        x_RGB_3,x_T_3 = RGB, T
        #print ('Block5------------55555',RGB.shape)         #[1, 512, 16, 16]
        
        # RGB_tp = torch.cat((x_RGB_3,x_T_prompt3),dim=1)
        # T_rp = torch.cat((x_T_3,x_RGB_prompt3),dim=1)
        # #print ('Block6------------66666',x.shape)
        # RGB = self.conv_rgb(RGB_tp)
        # T = self.conv_t(T_rp)
        #----------------
        x = self.fuse_weight_1 * RGB + self.fuse_weight_2 * T
        x = self.drop(x)

        #----------------
        #-----------------------------------
        prompt1 = (x_RGB_1,x_T_1,x_RGB_prompt1,x_T_prompt1)
        prompt2 = (x_RGB_2,x_T_2,x_RGB_prompt2,x_T_prompt2)
        prompt3 = (x_RGB_3,x_T_3,x_RGB_prompt3,x_T_prompt3) 
        
        #----------------------------------
    
        #-------------
        #print ('111111111111111',x.shape)
        #x = self.conv2d_x(x)
        #----------------
        x = F.interpolate(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x),prompt1,prompt2,prompt3

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
        
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

        self.cross_channels = cross_channels
        
        self.Flag = True
        
        if len(cfg) > 3:
            self.Encoder = Encoder(dim=cross_channels,num_heads=4,num_x_layers = 1,vit_patch_size=vit_img_size_)
            self.Flag = False
       

    def forward(self, RGB, T):
        if (self.Flag):
            
            RGB = self.rgb_conv(RGB)
            T = self.t_conv(T)
            
            return RGB, T
        
        else:
            if type(RGB) != list:
                RGB = self.rgb_conv(RGB)
                T = self.t_conv(T)

                RGB_att,T_att,x_RGB_prompt,x_T_prompt = self.Encoder(RGB,T)
                #---------------------
                RGB_att = self.drop1(RGB_att)
                T_att = self.drop2(T_att)
                #----------------------
                RGB = RGB + RGB_att
                T = T + T_att
                
                return RGB, T, x_RGB_prompt,x_T_prompt
            else:
                
                RGB_,x_RGB_prompt = RGB[0],RGB[1]
                T_,x_T_prompt = T[0],T[1]

                x_RGB_prompt_v = self.rgb_conv(x_RGB_prompt)
                x_T_prompt_v = self.t_conv(x_T_prompt)

                RGB_ = self.rgb_conv(RGB_)
                T_ = self.t_conv(T_)

                RGB_att,T_att,x_RGB_prompt,x_T_prompt = self.Encoder([RGB_,x_RGB_prompt_v],[T_,x_T_prompt_v])
                
                #---------------------
                RGB_att = self.drop1(RGB_att)
                T_att = self.drop2(T_att)

                x_RGB_prompt = self.drop1(x_RGB_prompt)
                x_T_prompt = self.drop2(x_T_prompt)
                #----------------------
                RGB = RGB_ + RGB_att
                T = T_+ T_att

                x_RGB_prompt = x_RGB_prompt + x_RGB_prompt_v
                x_T_prompt = x_T_prompt + x_T_prompt_v
                return RGB, T, x_RGB_prompt,x_T_prompt
        #---------drop

        



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
