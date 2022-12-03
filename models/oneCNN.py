import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch_dct as dct
import numpy as np
import math
from layers.dctnet import dct_channel_block,dct
        
class Model(nn.Module):#2022.11.7修改前，这个Model能跑通#CNN
    def __init__(self,configs):
        super(Model, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel_num = configs.enc_in
        self.conv1 = nn.Conv1d(in_channels=self.channel_num,out_channels=self.channel_num,kernel_size=5,stride=1,padding=2) #输入通道数和输出通道数应该一样
        # input = torch.randn(32,7,96)
        # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
        self.dct_layer=dct_channel_block(self.seq_len)
        self.linear = nn.Linear(self.seq_len,self.pred_len)
        # self.dct_norm = nn.LayerNorm([7], eps=1e-6)#作为模块一般normal channel效果好点



    def forward(self, x):
        # print("x.shape:",x.shape)
        x = x.permute(0,2,1) # (B，L,C)=》(B,C,L)#forL and 1-d conv
        # b, c, l = x.size() # (B,C,L)
        
        out = self.conv1(x) #b,c,l
        out = self.linear(x)#b,c,l
        # print(out.size())
        # out  = self.dct_layer(out)#加入dct模块，mse降低0.12个点
        # out = self.linear(x)#b,c,l
        # x = x+mid
        # x = x.permute(0,2,1) 
        # x = self.dct_norm(x) #norm 144
        # x = x.permute(0,2,1) 
        return  (out).permute(0,2,1)#b,l,c

