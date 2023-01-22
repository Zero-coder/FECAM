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
#         # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         # x = self.Linear(x.permute(0,2,1))
#         x = x.permute(0,2,1) # (B，L,C)=》(B,C,L)
#         b, c, l = x.size() # (B,C,L)
#         # y = self.avg_pool(x) # (B,C,L) 通过avg=》 (B,C,1)
#         # print("y",y.shape)
#         y = self.avg_pool(x).view(b, c) # (B,C,L) 通过avg=》 (B,C,1)
#         # print("y",y.shape)
#         #为了丢给Linear学习，需要view把数据展平开
#         # y = self.fc(y).view(b, c, 96)
        
#         y = self.fc(y).view(b,c,1)
#         # f_weight_np = y.cpu().detach().numpy()
#         z = self.Linear_More_1(x*y+x)
        
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
class Model(nn.Module):#2022.11.7修改前，这个Model能跑通#forMultivariate
        
    def __init__(self,configs,channel=96,ratio=1):#channel针对ili数据集应该改成36 channel=input_length
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
        # self.dct_norm = nn.LayerNorm([self.channel_num], eps=1e-6)#作为模块一般normal channel效果好点 for traffic
        self.dct_norm = nn.LayerNorm(self.seq_len, eps=1e-6)#
        # self.my_layer_norm = nn.LayerNorm([96], eps=1e-6)
    def forward(self, x):
        x = x.permute(0,2,1) # (B，L,C)=》(B,C,L)#forL

        
       
        b, c, l = x.size() # (B,C,L)
        list = []
        
        for i in range(c):#i represent channel ，分别对channel的数据做dct
            freq=dct.dct(x[:,i,:])     #dct
            # freq=torch.fft.fft(x[:,i,:])#fft,rfft的输出长度不与输入对齐 #fft结果不能通过linear

            # print("freq-shape:",freq.shape)
            list.append(freq)
            ##把dct结果进行拼接，再进行频率特征学习
        
        stack_dct=torch.stack(list,dim=1) 
        stack_dct = torch.tensor(stack_dct)#(B，L,C)
        
        stack_dct = self.dct_norm(stack_dct)#matters for traffic
        f_weight = self.fc(stack_dct)
        f_weight = self.dct_norm(f_weight)#matters for traffic
        
        #wo-dct
        # f_weight = self.dct_norm(x)
        # f_weight = self.dct_norm(f_weight)#matters for traffic
        # f_weight = self.fc(f_weight)
        # f_weight = self.dct_norm(f_weight)#matters for traffic
        # list_inverse = []
        # for i in range(c):
        #     freq_inverse=dct.idct(f_weight[:,i,:])     #dct
        #     # freq=torch.fft.fft(x[:,i,:])#fft,rfft的输出长度不与输入对齐 #fft结果不能通过linear

        #     # print("freq-shape:",freq.shape)
        #     list_inverse.append(freq_inverse)
        #     ##把dct结果进行拼接，再进行频率特征学习
        # stack_dct_inverse=torch.stack(list_inverse,dim=1) 
        # stack_dct_inverse = torch.tensor(stack_dct_inverse)#(B，L,C)
        # # f_weight = self.fc(x)
        # stack_dct_inverse = self.dct_norm(stack_dct_inverse)#matters for traffic
        # f_weight_inverse = self.fc_inverse(stack_dct_inverse)
        # f_weight_inverse = self.dct_norm(f_weight_inverse)#matters for traffic

        #可视化这个频率张量 generalized tensor
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
        
        # result = result + (1)*torch.mean(result)# for ill 增对数据集而设定的先验知识，可以后续有这方面的思考
        # result_1 = self.Linear_1(x)
        # result = result + result_1
        # result = self.my_layer_norm(result)

        return  result.permute(0,2,1)
        # return  result
        
# class Model(nn.Module):#2022.11.7修改前，这个Model能跑通#forMultivariate
#     def __init__(self,configs,channel=96,ratio=1):#channel针对ili数据集应该改成36 channel=input_length
#         super(Model, self).__init__()
#         # self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.fc = nn.Sequential(
#                 nn.Linear(channel, channel*4, bias=False),
#                 nn.Dropout(p=0.1),
#                 nn.ReLU(inplace=True),
#                 nn.Linear( channel*4, channel, bias=False),
#                 nn.Sigmoid()
#         )
#         # self.fc_plot = nn.Linear(channel, channel, bias=False)
#         self.mid_Linear = nn.Linear(self.seq_len, self.seq_len)

#         self.Linear = nn.Linear(self.seq_len, self.pred_len)
#         self.dct_norm = nn.LayerNorm([7], eps=1e-6)#作为模块一般normal channel效果好点 for traffic
        
#         # self.my_layer_norm = nn.LayerNorm([96], eps=1e-6)
#     def forward(self, x):
#         x = x.permute(0,2,1) # (B，L,C)=》(B,C,L)#forL

        
#         # x_t = self.mid_Linear(x)
#         b, c, l = x.size() # (B,C,L)
#         list = []
        
#         for i in range(c):#i represent channel ，分别对channel的数据做dct
#             freq=dct.dct(x[:,i,:])     #dct
#             # freq=torch.fft.fft(x[:,i,:])#fft,rfft的输出长度不与输入对齐 #fft结果不能通过linear

#             # print("freq-shape:",freq.shape)
#             list.append(freq)
#             ##把dct结果进行拼接，再进行频率特征学习
        
        
      
#         stack_dct=torch.stack(list,dim=1) 
#         stack_dct = torch.tensor(stack_dct)#(B，L,C)
#         f_weight = self.fc(stack_dct)

        
#         # f_weight = self.dct_norm(f_weight.permute(0,2,1))#matters for traffic
#         # result = self.Linear(x *(f_weight.permute(0,2,1)))#forL

#         result = self.Linear(x *(f_weight))#forL

        
#         # result = self.my_layer_norm(result)

#         return  result.permute(0,2,1)
        

