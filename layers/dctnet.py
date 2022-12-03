from distutils.command.config import config
import torch.nn as nn
import math
import numpy as np
import torch
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim = (-d))
        r = torch.stack((t.real, t.imag), -1)
        return r
    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:,:,0], x[:,:,1]), dim = (-d))
        return t.real

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    

    return V


# class dct_channel_block(nn.Module):
#     def __init__(self, channel=512, ratio=1):
#         super(dct_channel_block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
#         self.fc = nn.Sequential(
#                 nn.Linear(channel, channel // 4, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(channel //4, channel, bias=False),
#                 nn.Sigmoid()
#         )

#     def forward(self, x):
#         # b, c, l = x.size() # (B,C,L)
#         # y = self.avg_pool(x) # (B,C,L) 通过avg=》 (B,C,1)
#         # print("y",y.shape)
#         x = x.permute(0,2,1)
#         b, c, l = x.size() 
#         y = self.avg_pool(x).view(b, c) # (B,C,L) 通过avg=》 (B,C,1)
#         # print("y",y.shape)
#         #为了丢给Linear学习，需要view把数据展平开
#         # y = self.fc(y).view(b, c, 96)

#         y = self.fc(y).view(b,c,1)
#         # print("y",y.shape)
#         # return x * y
#         return (x*y).permute(0,2,1)
class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super(dct_channel_block, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
        self.fc = nn.Sequential(
                nn.Linear(channel, channel*2, bias=False),
                nn.Dropout(p=0.1),
                nn.ReLU(inplace=True),
                nn.Linear( channel*2, channel, bias=False),
                nn.Sigmoid()
        )
        # self.dct_norm = nn.LayerNorm([512], eps=1e-6)#作为模块一般normal channel效果好点 for traffic
  
        self.dct_norm = nn.LayerNorm([96], eps=1e-6)#for lstm on length-wise
        # self.dct_norm = nn.LayerNorm([36], eps=1e-6)#for lstm on length-wise on ill with input =36


    def forward(self, x):
        b, c, l = x.size() # (B,C,L) (32,96,512)
        # y = self.avg_pool(x) # (B,C,L) 通过avg=》 (B,C,1)
        
        # y = self.avg_pool(x).view(b, c) # (B,C,L) 通过avg=》 (B,C,1)
        # print("y",y.shape)
        #为了丢给Linear学习，需要view把数据展平开
        # y = self.fc(y).view(b, c, 96)
        list = []
        for i in range(c):#i represent channel ，分别对channel的数据做dct
            freq=dct(x[:,i,:])     
            # print("freq-shape:",freq.shape)
            list.append(freq)
            ##把dct结果进行拼接，再进行频率特征学习
        

        stack_dct=torch.stack(list,dim=1)
        stack_dct = torch.tensor(stack_dct)
        '''
        for traffic mission:f_weight = self.dct_norm(f_weight.permute(0,2,1))#matters for traffic datasets
        '''
        
        lr_weight = self.dct_norm(stack_dct) #不一定要,lstm用的时候不加
        lr_weight = self.fc(stack_dct)
        lr_weight = self.dct_norm(lr_weight) #不一定要,lstm用的时候不加
        
        # print("lr_weight",lr_weight.shape)
        return x *lr_weight #result


if __name__ == '__main__':
    
    tensor = torch.rand(8,7,96)
    dct_model = dct_channel_block()
    result = dct_model.forward(tensor) 
    print("result.shape:",result.shape)

#Informer channel*2
#

