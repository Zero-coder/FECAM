import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch_dct as dct

import numpy as np
import math
# class Model(nn.Module):SENET for ETTmx

#     """
#     Just one Linear layer
#     """
#     def __init__(self,configs,channel=7,ratio=1):
#         super(Model, self).__init__()

#         self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
#         self.fc = nn.Sequential(
#                 nn.Linear(7,14, bias=False),
#                 nn.Dropout(p=0.1),
#                 nn.ReLU(inplace=True) ,
#                 nn.Linear(14,7, bias=False),
#                 nn.Sigmoid()
#         )
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len

#         self.Linear_More_1 = nn.Linear(self.seq_len,self.pred_len * 2)
#         self.Linear_More_2 = nn.Linear(self.pred_len*2,self.pred_len)
#         self.relu = nn.ReLU()
#         self.gelu = nn.GELU()    

#         self.drop = nn.Dropout(p=0.1)
#         # Use this line if you want to visualize the weights
#        
#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#   
#         x = x.permute(0,2,1) # (B，L,C)->(B,C,L)
#         b, c, l = x.size() # (B,C,L)
#         y = self.avg_pool(x).view(b, c) # (B,C,L) 
        
        
#         # np.save('f_weight.npy', f_weight_np)
# #         # np.save('%d f_weight.npy' %epoch, f_weight_np)
#         # print("y",y.shape)
#         # return (x * y).permute(0,2,1)
#         return (z).permute(0,2,1)

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias
class Model(nn.Module):
        
    def __init__(self,configs,channel=96,ratio=1):
        super(Model, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel_num = configs.enc_in
        self.fc = nn.Sequential(
                nn.Linear(channel, channel*2, bias=False),
                nn.Dropout(p=0.1),
                nn.ReLU(inplace=True),
                nn.Linear( channel*2, channel, bias=False),
                nn.Sigmoid()
        )
        self.fc_inverse = nn.Sequential(
            nn.Linear(channel, channel//2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear( channel//2, channel, bias=False),
            nn.Sigmoid()
        )
        # self.fc_plot = nn.Linear(channel, channel, bias=False)
        self.mid_Linear = nn.Linear(self.seq_len, self.seq_len)

        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_1 = nn.Linear(self.seq_len, self.pred_len)
        # self.dct_norm = nn.LayerNorm([self.channel_num], eps=1e-6)
        self.dct_norm = nn.LayerNorm(self.seq_len, eps=1e-6)
        # self.my_layer_norm = nn.LayerNorm([96], eps=1e-6)
    def forward(self, x):
        x = x.permute(0,2,1) # (B，L,C)=》(B,C,L)#forL

        
       
        b, c, l = x.size() # (B,C,L)
        list = []
        
        for i in range(c):#i represent channel 
            freq=dct.dct(x[:,i,:])     #dct
            # print("freq-shape:",freq.shape)
            list.append(freq)
         
        
        stack_dct=torch.stack(list,dim=1) 
        stack_dct = torch.tensor(stack_dct)#(B，L,C)
        
        stack_dct = self.dct_norm(stack_dct)#matters for traffic
        f_weight = self.fc(stack_dct)
        f_weight = self.dct_norm(f_weight)#matters for traffic
        


        #visualization for fecam tensor
        f_weight_cpu = f_weight
        
        f_weight_np = f_weight_cpu.cpu().detach().numpy()
        
        np.save('f_weight_weather_wf.npy', f_weight_np)
        # np.save('%d f_weight.npy' %epoch, f_weight_np)


        




        # f_weight = self.dct_norm(f_weight.permute(0,2,1))#matters for traffic
        # result = self.Linear(x)#forL
        
        # f_weight_np = result.cpu().detach().numpy()
        
        # np.save('f_weight.npy', f_weight_np)
        # x = x.permute(0,2,1)
        # result = self.Linear((x *(f_weight_inverse)))#forL 
        result = self.Linear((x *(f_weight)))#forL
        return  result.permute(0,2,1)
        
        


