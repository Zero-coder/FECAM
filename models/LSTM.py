from pip import main
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch_dct as dct
import numpy as np
import math
from layers.dctnet import dct_channel_block,dct
# from dctnet import dct_channel_block,dct #for parameters calc
import argparse
from fvcore.nn import FlopCountAnalysis,parameter_count_table  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):#2022.11.7修改前，这个Model能跑通
    def __init__(self,configs,input_size=None,hidden_size=16, output_size=None,batch_size=None,num_layers=2):#input_size与output_size是通道数
        super(Model, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_size = configs.enc_in #channel
        self.hidden_size = hidden_size #输出维度 也就是输出通道
        self.num_layers = num_layers
        self.output_size = configs.enc_in #channel #输出个数
        self.num_directions = 1 # 单向LSTM
        self.batch_size = configs.batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)#通道数对齐层
        self.linear_out_len = nn.Linear(self.seq_len, self.pred_len)#输出长度对齐层
        # self.linear = nn.Linear()
        #add dce-block
        self.dct_layer=dct_channel_block(configs.seq_len)
        self.dct_norm = nn.LayerNorm([configs.enc_in], eps=1e-6)#作为模块一般normal channel效果好点
        # self.dct_norm = nn.LayerNorm([512], eps=1e-6)#作为模块一般normal channel效果好点
    def forward(self, x):
        # x = x.permute(0,2,1) # (B，L,C)=》(B,C,L)#forL
        # b, c, l = x.size() # (B,C,L)

        # batch_size, seq_len = x.shape[0], x.shape[1]
        # h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x, (h_0, c_0)) # output(5, 30, 64)
        # print("output.shape:",output.shape)#result.shape: torch.Size([8, 96, 8])
        result = self.linear(output)  # (B，L,C)
        # output.shape: torch.Size([8, 96, 16])#16是hidden_size
        # result.shape: torch.Size([8, 96, 8])#8是为了符合通道数对齐，exchange_rate有8个变量
        '''
        dct
        '''
        result = self.dct_layer(result.permute(0,2,1))#加入dct模块，mse降低0.12个点
        result_len =  self.linear_out_len(result)#为了输出长度对齐 (8,8,96)
        '''
        dct
        '''
        # result_len =  self.linear_out_len(result.permute(0,2,1))#为了输出长度对齐 (8,8,96)
            

        # print("result.shape:",result.shape)#result.shape: torch.Size([8, 96, 8])
        # result = result.permute(0,2,1)#(B，L,C)=》(B,C,L)
        # result = self.dct_layer(result_len)#加入dct模块，mse降低0.12个点
        # result = result.permute(0,2,1)#(B，C,L)=》(B,L,C)

        return  result_len.permute(0,2,1)
# lstm = LSTM(input_size=7,hidden_size=64, output_size=7,batch_size=8,num_layers=5).to(device)
# tensor = torch.rand(8, 96, 7).to(device)
# result = lstm(tensor)
# print("result.shape:",result.shape)
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()
    model = Model(args)
    print(parameter_count_table(model))