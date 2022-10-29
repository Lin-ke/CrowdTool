import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class patchembed(nn.Module):
   
    def __init__(self, img_size=8, patch_size=2, in_c=256, embed_dim=768):
        super(patchembed, self).__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size//patch_size, img_size//patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
       
        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        
       
        #norm_layer = nn.LayerNorm(dim, eps=1e-6)
        #self.norm = norm_layer(embed_dim)
        

    def forward(self, inputs):
       
        B, C, H, W = inputs.shape

        #assert H==self.img_size[0] and W==self.img_size[1], 'input shape does not match 224*224'
        x = self.proj(inputs)  
        _,_, H_,W_ = x.shape
        x = x.flatten(start_dim=2, end_dim=-1) 
        x = x.transpose(1, 2) 
        #H_,W_ = H//self.patch_size[0], W//self.patch_size[0]
        #x = self.norm(x)
        
        return x,(H_,W_)


class MLP(nn.Module):
    def __init__(self, in_features=768, out_features=None, drop=0.1):
        super(MLP, self).__init__()
        hidden_features = int(in_features * 4)
        out_features = out_features 
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=in_features)

        self.drop = nn.Dropout(drop)  
        self.layer_norm = nn.LayerNorm(in_features, eps=1e-6)

    def forward(self, inputs):
        
        # [b,197,768]==>[b,197,3072]
        
        B,C,H,W = inputs.shape
        inputs = inputs.reshape(B,C,H*W).permute(0,2,1)
        residual = inputs 
        x = self.layer_norm(inputs)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        
        # [b,197,3072]==>[b,197,768]
        x = self.fc2(x)
        x = self.drop(x)
        x = x + residual
        
        return x
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=4, qkv_bias=False, atten_drop_ratio=0.1, proj_drop_ratio=0.1):
        super(MultiHeadAttention, self).__init__()
        
        
        self.num_heads = num_heads  
        
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
      
        
        self.qkv = nn.Linear(in_features=dim, out_features=dim*3, bias=qkv_bias)
        self.qkv1 = nn.Linear(in_features=dim, out_features=dim*3, bias=qkv_bias)
        
        self.conv2d_RGB = nn.Sequential(
                     nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3,3), stride=(1, 1), padding=(1, 1), bias=False),
                     nn.ReLU(True),
                     nn.Dropout(0.5),
                     nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3,3), stride=(1, 1), padding=(1, 1), bias=False),
                     nn.ReLU(True),
                     nn.Dropout(0.5)
                     )
        
        self.conv2d_T = nn.Sequential(
                     nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3,3), stride=(1, 1), padding=(1, 1), bias=False),
                     nn.ReLU(True),
                     nn.Dropout(0.5),
                     nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3,3), stride=(1, 1), padding=(1, 1), bias=False),
                     nn.ReLU(True),
                     nn.Dropout(0.5)
                     )
       
        self.conv1d = nn.Sequential(
                     nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=(1,1), stride=(1, 1), bias=False)                    
                     )
        self.conv1d2 = nn.Sequential(
                     nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=(1,1), stride=(1, 1), bias=False)                    
                     )
        
        self.atten_drop1 = nn.Dropout(atten_drop_ratio)
        self.atten_drop2 = nn.Dropout(atten_drop_ratio)
        
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        #self.layer_norm3 = nn.LayerNorm(dim, eps=1e-6)
        #self.layer_norm4 = nn.LayerNorm(dim, eps=1e-6)
        
        
        self.proj1 = nn.Linear(in_features=dim, out_features=dim)
        self.proj2 = nn.Linear(in_features=dim, out_features=dim)
        
        self.proj_drop1 = nn.Dropout(proj_drop_ratio)
        self.proj_drop2 = nn.Dropout(proj_drop_ratio)
        #self.x_RGB_T_drop = nn.Dropout(0.5)
    
    
    def forward(self, inputs_R_ori,inputs_T_ori,hw):
    
        B, N, C = inputs_R_ori.shape
        HW = hw
        
        #---------------------------conv_self_attention----------
        #print ('Cross-----------------11111111111',inputs_R.shape)
        
        #---------------test 
        inputs_R = self.layer_norm1(inputs_R_ori)
        inputs_T = self.layer_norm2(inputs_T_ori)
        
        inputs_R_cnn = inputs_R.permute(0,2,1).reshape(B,C,HW[0],HW[1])
        inputs_T_cnn = inputs_T.permute(0,2,1).reshape(B,C,HW[0],HW[1]) 
        
        #------------------
        
        inputs_R_cnn_residual = inputs_R_ori.permute(0,2,1).reshape(B,C,HW[0],HW[1])
        inputs_T_cnn_residual = inputs_T_ori.permute(0,2,1).reshape(B,C,HW[0],HW[1]) 
        
        x_RGB = self.conv2d_RGB(inputs_R_cnn)
        #print ('Cross-----------------22222222222',x_RGB.shape)
        x_T = self.conv2d_T(inputs_T_cnn) 
        #print ('Cross-----------------3333333333',x_T.shape)
        
        x_RGB_T = torch.sigmoid(x_T) * x_RGB + x_RGB   #x_RGB_T = torch.sigmoid(x_T) * x_RGB + inputs_R_cnn_residual     #x_RGB1_T = F.sigmoid(x_T) * x_T + x_RGB
               
        x_T_RGB = torch.sigmoid(x_RGB) * x_T + x_T  #x_T_RGB = torch.sigmoid(x_RGB) * x_T + inputs_T_cnn_residual

        #---------------------------self_attention-------https://blog.csdn.net/dgvv4/article/details/125184340-------- 
        #---------------------------https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
        #---------------------------https://github.com/airsplay/lxmert/blob/master/src/lxrt/modeling.py
         
        #inputs_R_ori = inputs_R_ori.reshape(B,H*W,C)
        #inputs_T_ori = inputs_T_ori.reshape(B,H*W,C)
        
        
        #inputs_R = self.layer_norm1(inputs_R_ori)
        #inputs_T = self.layer_norm2(inputs_T_ori)
 
        qkv = self.qkv(inputs_R)       #RGB
        
        qkv1 = self.qkv1(inputs_T)     # T
       
 
        qkv = qkv.reshape(B, N, 3, self.num_heads, C//self.num_heads)
        qkv1 = qkv1.reshape(B, N, 3, self.num_heads, C//self.num_heads)
       
        
        qkv = qkv.permute(2,0,3,1,4)
        qkv1 = qkv1.permute(2,0,3,1,4)
        
        #----------------------RGB---V-------------------------
        q = qkv[0]      #RGB
        k, v = qkv1[1], qkv1[2]    # T           
        R_atten = (q @ k.transpose(-2,-1)) * self.scale       
        R_atten = R_atten.softmax(dim=-1)     
        R_atten = self.atten_drop1(R_atten)
        R_x = R_atten @ v      
        R_x = R_x.transpose(1,2)      
        R_x = R_x.reshape(B,N,C)
                          
        R_x = self.proj1(R_x)     
        R_x = self.proj_drop1(R_x) 
        #R_x = self.layer_norm3(R_x + inputs_R_ori)
        R_x = R_x + inputs_R_ori 
          
        R_x = R_x.reshape(B,C,HW[0],HW[1])      
        R_x = torch.cat((R_x,x_RGB_T),dim=1)
        R_x = self.conv1d(R_x)
        
        #R_x = R_x.reshape(B,C,HW[0],HW[1]).permute(0,2,1)
        #R_x = R_x + inputs_R_ori
        #R_x = R_x.permute(0,2,1).reshape(B,C,HW[0],HW[1])
        #------------------------TTTT-----------------------      
        q2 = qkv1[0]   # T
        k2, v2 = qkv[1], qkv[2]  #RGB          
        T_atten = (q2 @ k2.transpose(-2,-1)) * self.scale        
        T_atten = T_atten.softmax(dim=-1)       
        T_atten = self.atten_drop2(T_atten) 
        T_x = T_atten @ v2       
        T_x = T_x.transpose(1,2)     
        T_x = T_x.reshape(B,N,C)      
        
        T_x = self.proj2(T_x)       
        T_x = self.proj_drop2(T_x)  
        #T_x = self.layer_norm4(T_x + inputs_T_ori)  
        T_x = T_x + inputs_T_ori                   
        T_x = T_x.reshape(B,C,HW[0],HW[1])
        T_x = torch.cat((T_x,x_T_RGB),dim=1)
        T_x = self.conv1d2(T_x)
        
        
        #T_x = T_x.reshape(B,C,HW[0],HW[1]).permute(0,2,1)
        #T_x = T_x + inputs_T_ori 
        #T_x = T_x.permute(0,2,1).reshape(B,C,HW[0],HW[1])
        
        return R_x,T_x

class Encoderlayer(nn.Module):
    def __init__(self, dim=256, num_heads=4,vit_patch_size=[32,2]):
        super(Encoderlayer, self).__init__()
        # The cross-attention Layer
        
        
        
        self.patchembedRgb = patchembed(img_size = vit_patch_size[0], patch_size=vit_patch_size[1], in_c=dim, embed_dim=768)
        self.patchembedT = patchembed(img_size = vit_patch_size[0], patch_size=vit_patch_size[1], in_c=dim, embed_dim=768)
        
        self.cross_att = MultiHeadAttention(dim = 768, num_heads=4)
        
        self.FFN_RGB = MLP(in_features = 768, out_features=dim)
        self.FFN_T = MLP(in_features = 768, out_features=dim)
        self.conv1d1 = nn.Sequential(
                     nn.Conv2d(in_channels=768, out_channels=dim, kernel_size=(1,1), stride=(1, 1), bias=False)                    
                     )
        self.conv1d2 = nn.Sequential(
                     nn.Conv2d(in_channels=768, out_channels=dim, kernel_size=(1,1), stride=(1, 1), bias=False)                    
                     )
        
        self.flag = False
        if vit_patch_size[1] > 1:
            self.flag = True
        
    '''
    def cross_att(self, x_RGB, x_T):
        # Cross Attention
        RGB_att_output,T_att_output = self.visual_attention(x_RGB , x_T)
        
        return RGB_att_output,T_att_output
   '''


    def forward(self, x_RGB, x_T):
        
        _,_,H,W = x_RGB.shape
        x_RGB,hw = self.patchembedRgb(x_RGB)
        
        x_T,hw = self.patchembedT(x_T)
        
        x_RGB, x_T = self.cross_att(x_RGB, x_T,hw)
        
        x_RGB = self.FFN_RGB(x_RGB)
        x_T = self.FFN_T(x_T)
        
        B,N,C = x_T.shape
        
        x_RGB = x_RGB.permute(0,2,1).reshape(B,C,hw[0],hw[1])
        x_T = x_T.permute(0,2,1).reshape(B,C,hw[0],hw[1])
        
        x_RGB = self.conv1d1(x_RGB) 
        x_T = self.conv1d2(x_T)
        if (self.flag):
            x_RGB = F.interpolate(x_RGB, size=(H,W))
            x_T = F.interpolate(x_T, size=(H,W))
        #print ('11111111111111111111111111111',x_RGB.shape)
        
        

        return x_RGB, x_T

class Encoder(nn.Module):
    def __init__(self, dim=256, num_heads=4,num_x_layers = 1,vit_patch_size=[32,2]):
        super().__init__()

        # Obj-level image embedding layer
       

        # Number of layers
        
        self.num_x_layers = num_x_layers   
       
        self.x_layers = nn.ModuleList(
                            [Encoderlayer(dim, num_heads,vit_patch_size) for _ in range(self.num_x_layers)]
                                 )
        
       

    def forward(self, x_RGB, x_T):

        x_RGB_att = x_RGB
        x_T_att = x_T
        # Run cross-modality layers
        for layer_module in self.x_layers:
            x_RGB_att, x_T_att = layer_module(x_RGB_att, x_T_att)

        return x_RGB_att, x_T_att
