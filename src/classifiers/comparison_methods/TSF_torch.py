"The implementation of article 'Triple Spectral Fusion For Sensor-Based Human Activity Recognition' (TSF)"

import torch
torch.cuda.current_device()
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
# import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import time
from utils.utils import *
import os
from torch.nn.utils import weight_norm
import pywt
from torch.autograd import Function
from thop import profile
from thop import clever_format
import scipy.io as scio
import dgl
from dgl import function as fn

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

class DWT_1D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'db4',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False, INFERENCE_DEVICE='TEST_CUDA'):
        
        super(DWT_1D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        self.INFERENCE_DEVICE = INFERENCE_DEVICE
        if not self.trainable:
            assert self.kernel_size == None, 'if trainable = False, set kernel_size as None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0, 'the value of groups should be divisible by in_channels'
        self.stride = stride
        assert self.stride == 2, 'stride should only set as 2'
        self.wavename = wavename
        self.pad_type = pad_type
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, 'kernel_size cannot be smaller than the length of adopted wavelet'
        a = (self.kernel_size - length_band) // 2 
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b                
        self.filt_low  = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b]  = band_low           # put wavelet in the middle of the kernel
        self.filt_high[a:b] = band_high

    def initialization(self):
        
        self.filter_low = self.filt_low[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        self.filter_high = self.filt_high[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        if self.INFERENCE_DEVICE == 'TEST_CUDA':
            self.filter_low = self.filter_low.cuda()
            self.filter_high = self.filter_high.cuda()
        if self.trainable:
            self.filter_low = nn.Parameter(self.filter_low)
            self.filter_high = nn.Parameter(self.filter_high)

        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 3
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv1d(input, self.filter_low, stride = self.stride, groups = self.groups), \
               F.conv1d(input, self.filter_high, stride = self.stride, groups = self.groups)

class DWT_2D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'db4',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False, INFERENCE_DEVICE='TEST_CUDA'):
        super(DWT_2D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        self.INFERENCE_DEVICE = INFERENCE_DEVICE
        if not self.trainable:
            assert self.kernel_size == None, 'if trainable = False, set kernel_size as None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0, 'the value of groups should be divisible by in_channels'
        self.stride = stride
        assert self.stride == 2, 'stride should only set as 2'
        self.wavename = wavename
        self.pad_type = pad_type
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, 'kernel_size cannot be smaller than the length of adopted wavelet'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low  = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b]  = band_low           # put wavelet in the middle of the kernel
        self.filt_high[a:b] = band_high

    def initialization(self):
        
        self.filter_low  = self.filt_low[None, None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_high = self.filt_high[None, None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        if self.INFERENCE_DEVICE == 'TEST_CUDA':
            self.filter_low = self.filter_low.cuda()
            self.filter_high = self.filter_high.cuda()
        if self.trainable:
            self.filter_low = nn.Parameter(self.filter_low)
            self.filter_high = nn.Parameter(self.filter_high)

        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1, 0, 0]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2, 0, 0]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 4
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv2d(input, self.filter_low, stride = [1,self.stride], groups = self.groups), \
               F.conv2d(input, self.filter_high, stride = [1,self.stride], groups = self.groups)

class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()
    
class IMU_Fusion_Block(nn.Module): # TCN no dialation
    def __init__(self, input_2Dfeature_channel, input_channel, 
                 feature_channel, kernel_size_grav,
                 scale_num, dataset_name):
        super(IMU_Fusion_Block, self).__init__()
        
        self.scale_num         = scale_num
        self.input_channel     = input_channel
        self.tcn_grav_convs    = []
        self.tcn_gyro_convs    = []
        self.tcn_acc_convs     = []
        
        for i in range(self.scale_num):
            
            kernel_size_gyro  = kernel_size_grav - 1
            kernel_size_acc   = kernel_size_grav
            self.kernel_size_gyro = kernel_size_gyro
            
            tcn_grav = nn.Sequential(
                nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                          (1,kernel_size_grav), 1, (0,kernel_size_grav//2), 
                          dilation=1),
                nn.BatchNorm2d(feature_channel),
                nn.ReLU()
                )
            
            
            if kernel_size_gyro == 1:
                tcn_gyro = nn.Sequential(
                    nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                                          (1,1), 1, (0,0), 
                                          dilation=1),
                    nn.BatchNorm2d(feature_channel),
                    nn.ReLU()
                    )
            else:
                tcn_gyro = nn.Sequential(
                    nn.ZeroPad2d(((kernel_size_gyro//2-1), kernel_size_gyro//2, 0, 0)),
                    nn.Conv2d(input_2Dfeature_channel, feature_channel, 
                              (1,kernel_size_gyro), 1, (0,0), 
                              dilation=1),
                    nn.BatchNorm2d(feature_channel),
                    nn.ReLU()
                    )
            
            tcn_acc = nn.Sequential(
                nn.Conv2d(input_2Dfeature_channel, feature_channel,
                          (1,kernel_size_acc), 1, (0,kernel_size_acc//2), 
                          dilation=1),
                nn.BatchNorm2d(feature_channel),
                nn.ReLU()
                )
            
            setattr(self, 'tcn_grav_convs%i' % i, tcn_grav)
            self.tcn_grav_convs.append(tcn_grav)
            setattr(self, 'tcn_gyro_convs%i' % i, tcn_gyro)
            self.tcn_gyro_convs.append(tcn_gyro)
            setattr(self, 'tcn_acc_convs%i' % i, tcn_acc)
            self.tcn_acc_convs.append(tcn_acc)
        
        self.attention = nn.Sequential(
                nn.Linear(3*feature_channel, 1),
                nn.PReLU(),
                )
        
    def forward(self, x):
        
        x_grav = x[:,:,0:3,:]
        x_gyro = x[:,:,3:6,:]
        x_acc  = x[:,:,6:9,:]
    
        for i in range(self.scale_num):
            
            out_grav = self.tcn_grav_convs[i](x_grav).unsqueeze(4)
            out_gyro = self.tcn_gyro_convs[i](x_gyro).unsqueeze(4)
            out_acc  = self.tcn_acc_convs[i](x_acc)
            
            if i == 0:
                out_attitude = torch.cat([out_grav, out_gyro], dim=4)
                out_dynamic  = out_acc
            else:
                out_attitude = torch.cat([out_attitude, out_grav], dim=4)
                out_attitude = torch.cat([out_attitude, out_gyro], dim=4)
                out_dynamic  = torch.cat([out_dynamic, out_acc], dim=2)
                
        # (batch_size, time_length, sensor_num*scale_num, 3(xyz), feature_chnnl)
        out_attitude = out_attitude.permute(0,3,4,2,1)
        # (batch_size, time_length, sensor_num*scale_num, 3(xyz)*feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2], -1)
        
        # time-step-wise sensor attention, sensor_attn:(batch_size, time_length, sensor_num*scale_num, 1)
        sensor_attn  = self.attention(out_attitude).squeeze(3)
        sensor_attn  = F.softmax(sensor_attn, dim=2).unsqueeze(-1)
        
        out_attitude = sensor_attn * out_attitude
        
        # cal norm
        norm_num     = torch.mean(sensor_attn.squeeze(-1), dim=1)
        norm_num     = torch.pow(norm_num, 2)
        norm_num     = torch.sqrt(torch.sum(norm_num, dim=1))
        norm_num     = (pow(self.scale_num,0.5)/norm_num).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        out_attitude = out_attitude * norm_num
        
        # (batch_size, time_length, sensor_num*scale_num, 3(xyz), feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2], 3, -1)
        # (batch_size, time_length, sensor_num*scale_num*3(xyz), feature_chnnl)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2]*3, -1)
        # (batch_size, feature_chnnl, sensor_num*scale_num*3(xyz), time_length)
        out_attitude = out_attitude.permute(0,3,2,1)
        
        # concatenate all the different scales
        out_attitude = torch.split(out_attitude, 6, dim=2)
        for j in range(len(out_attitude)):
            per_scale_attitude = torch.split(out_attitude[j], 3, dim=2)
            for k in range(len(per_scale_attitude)):
                if k == 0:
                    per_attitude   = per_scale_attitude[k]
                else:
                    per_attitude   = per_attitude + per_scale_attitude[k]
            if j == 0:
                all_attitude = per_attitude
            else:
                all_attitude = torch.cat([all_attitude, per_attitude], dim=2)
        
        out_attitude = all_attitude
        out          = torch.cat([out_attitude, out_dynamic], dim = 2)
        
        return out, sensor_attn

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x

class gumble_block_2D(nn.Module):
    def __init__(self, inchannel, outchannel, data_channel, data_length):
        super(gumble_block_2D, self).__init__()
        
        self.ch_mask_1  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannel, inchannel//2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(inchannel//2, outchannel, kernel_size=1),
            nn.PReLU(),
        )

        self.tau        = 1 # nn.Parameter(torch.tensor([1.]))
        self.outchannel = outchannel

    def _update_tau(self, tau):
        self.tau = tau

    def forward(self, x_low, x_high, test_flag):
        
        x   = torch.cat((x_low, x_high), dim=1)
        out = torch.cat((x_low.unsqueeze(1), x_high.unsqueeze(1)), dim=1)
        ch_mask_1  = self.ch_mask_1(x)

        ch_mask_1  = gumbel_softmax(ch_mask_1, dim=1, tau=self.tau).unsqueeze(-1) # [128, 2, 1, 1, 1]
        
        if test_flag == True:
            ch_mask_1      = torch.argmax(ch_mask_1, 1)
            ch_mask_1      = torch.cat([(1-ch_mask_1), ch_mask_1], dim=1).float().reshape(out.shape[0],-1,1,1,1)
        
        input_conv = torch.sum(out * ch_mask_1, dim=1)
        input_res  = torch.sum(out * (1 - ch_mask_1), dim=1)
        
        return input_conv, input_res, torch.argmax(ch_mask_1.squeeze(-1).squeeze(-1),1)

class gumble_block_1D(nn.Module):
    def __init__(self, inchannel, outchannel, data_length):
        super(gumble_block_1D, self).__init__()
        
        self.ch_mask_1  = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(inchannel, inchannel//4, kernel_size=1),
            nn.PReLU(),
            nn.Conv1d(inchannel//4, outchannel, kernel_size=1),
            nn.PReLU()
        )

        self.tau        = 1
        self.outchannel = outchannel

    def _update_tau(self, tau):
        self.tau = tau

    def forward(self, x_low, x_high, test_flag):
        
        x   = torch.cat((x_low, x_high), dim=1)
        
        out = torch.cat((x_low.unsqueeze(1), x_high.unsqueeze(1)), dim=1)
        ch_mask_1  = self.ch_mask_1(x)

        ch_mask_1  = gumbel_softmax(ch_mask_1, dim=1, tau=self.tau).unsqueeze(-1)
        
        if test_flag == True:
            ch_mask_1        = torch.argmax(ch_mask_1, 1)
            ch_mask_1        = torch.cat([(1-ch_mask_1), ch_mask_1], dim=1).float().reshape(out.shape[0],-1,1,1)
        
        input_conv = torch.sum(out * ch_mask_1, dim=1)
        input_res  = torch.sum(out * (1 - ch_mask_1), dim=1)
        
        return input_conv, input_res, torch.argmax(ch_mask_1.squeeze(-1),1)

class FALayer(nn.Module):
    def __init__(self, in_dim, dropout):
        super(FALayer, self).__init__()
        
        self.in_dim = in_dim
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(3*in_dim//2, 3*in_dim//2),
                                  nn.ReLU(),
                                  nn.Linear(3*in_dim//2, 1),
                                  nn.Tanh()
                                 )
        self.graph_act = nn.Tanh()

    def edge_applying(self, edges):
        
        h2 = edges.dst['h'] * edges.src['h']
        g  = self.gate(h2)
        
        e  = g
        e  = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, g, h):
        self.g            = g
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))
        return self.g.ndata['z'], self.g.edata['e']

class HeteGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, POS_NUM, layer_num=2):
        super(HeteGNN, self).__init__()
        
        self.pos_num   = POS_NUM
        self.layer_num = layer_num
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        
        self.layers   = nn.ModuleList()
        self.gate_res = nn.ModuleList()
        
        self.activation = nn.PReLU()
        self.BN_norm    = nn.BatchNorm1d(out_dim)
        self.LN_norm    = nn.LayerNorm(out_dim)
        
        for i in range(self.layer_num):
            self.layers.append(FALayer(hidden_dim, dropout))

        self.t1_posture = nn.Sequential(
                                        nn.Conv2d(in_channels=3*in_dim,
                                                  out_channels=3*hidden_dim,
                                                  kernel_size=1,
                                                  groups=3),
                                        nn.PReLU(),
                                        nn.Conv2d(in_channels=3*hidden_dim,
                                                  out_channels=3*hidden_dim//2,
                                                  kernel_size=1,
                                                  groups=3),
                                        )
        self.t1_motion  = nn.Sequential(
                                        nn.Conv2d(in_channels=3*in_dim,
                                                  out_channels=3*hidden_dim,
                                                  kernel_size=1,
                                                  groups=3),
                                        nn.PReLU(),
                                        nn.Conv2d(in_channels=3*hidden_dim,
                                                  out_channels=3*hidden_dim//2,
                                                  kernel_size=1,
                                                  groups=3),
                                        )
        
        self.t2 = nn.Linear(3*3*in_dim//2, out_dim)

    def forward(self, g, h):
        
        h_posture = h[:,:,:,0,:].permute(0,3,2,1)
        h_motion  = h[:,:,:,1,:].permute(0,3,2,1)
        h_posture = self.t1_posture(h_posture)
        h_motion  = self.t1_motion(h_motion)
        h_posture = h_posture.permute(0,3,2,1).unsqueeze(3)
        h_motion  = h_motion.permute(0,3,2,1).unsqueeze(3)
        h = torch.cat((h_posture, h_motion), axis=3)
        h = h.reshape(-1, h.shape[-1])
        raw = h
        for i in range(self.layer_num):
            
            h = self.layers[i](g, h)[0] + h

            if i == 0:
                hh = h
                ee = self.layers[i](g, h)[1]
            else:
                hh = torch.cat((hh, h), 1)
                ee = torch.cat((ee, self.layers[i](g, h)[1]), 0)
        
        h = torch.cat((raw, hh), 1)
        h = self.activation(self.BN_norm(self.LN_norm(self.t2(h))))
        h = self.dropout2(h)
        
        return h, ee

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.transpose(1,2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)

class SelfAttention(nn.Module):
    def __init__(self, k, heads = 8, drop_rate = 0):
        super(SelfAttention, self).__init__()
        self.k, self.heads = k, heads
        
        self.tokeys    = nn.Linear(k, k * heads, bias = False)
        self.toqueries = nn.Linear(k, k * heads, bias = False)
        self.tovalues  = nn.Linear(k, k * heads, bias = False)
        
        self.dropout_attention = nn.Dropout(drop_rate)
        
        self.unifyheads = nn.Linear(heads * k, k)
        
    def forward(self, x):
        
        b, t, k = x.size()
        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x).view(b, t, h, k)
        values  = self.tovalues(x).view(b, t, h, k)
        
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys    = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values  = values.transpose(1, 2).contiguous().view(b * h, t, k)
        
        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))
        
        dot  = torch.bmm(queries, keys.transpose(1,2))
        
        dot = F.softmax(dot, dim=2)
        dot = self.dropout_attention(dot)
        out = torch.bmm(dot, values).view(b, h, t, k)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h*k)
        
        return self.unifyheads(out) # (b, t, k)

class Transformer_1(nn.Module):
    def __init__(self, k, heads, drop_rate):
        super(Transformer_1, self).__init__()

        self.attention = SelfAttention(k, heads = heads, drop_rate = drop_rate)
        self.norm1     = nn.BatchNorm1d(k)

        self.mlp = nn.Sequential(
            nn.Conv1d(k, 4*k, 1, 1),
            nn.PReLU(),
            nn.Conv1d(4*k, k, 1, 1)
        )
        self.norm2              = nn.BatchNorm1d(k)
        self.dropout_forward    = nn.Dropout(drop_rate)

    def forward(self, x, x_high):
        
        attended = self.attention(x)
        attended = attended + x + x_high
        attended = attended.permute(0,2,1)
        
        x        = self.norm1(attended)
        
        feedforward = self.mlp(x)
        feedforward = feedforward + x
        
        return self.dropout_forward(self.norm2(feedforward))


class Transformer_2(nn.Module):
    def __init__(self, k, heads, drop_rate):
        super(Transformer_2, self).__init__()

        self.attention = SelfAttention(k, heads = heads, drop_rate = drop_rate)
        self.norm1     = nn.BatchNorm1d(k)

        self.mlp = nn.Sequential(
            nn.Conv1d(k, 4*k, 1, 1),
            nn.PReLU(),
            nn.Conv1d(4*k, k, 1, 1)
        )
        self.norm2                = nn.BatchNorm1d(k)
        self.dropout_forward      = nn.Dropout(drop_rate)

    def forward(self, x, x_high):
        
        attended = self.attention(x)
        attended = attended + x + x_high
        attended = attended.permute(0,2,1)
        
        x           = self.norm1(attended)
        feedforward = self.mlp(x)
        feedforward = feedforward + x
        
        return self.dropout_forward(self.norm2(feedforward))

class TSF(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
                 kernel_size_grav, kernel_size, scale_num, feature_channel_out,
                 multiheads, drop_rate, dataset_name, POS_NUM, data_length, train_size,
                 val_size, test_size, num_class, BATCH_SIZE, INFERENCE_DEVICE, test_split):
        
        super(TSF, self).__init__()
        
        self.feature_channel  = feature_channel
        self.POS_NUM          = POS_NUM
        self.feature_channel_out  = feature_channel_out
        self.BATCH_SIZE       = BATCH_SIZE
        self.test_split       = test_split
        
        self.IMU_fusion_blocks = []
        for i in range(POS_NUM):
            IMU_fusion_block   = IMU_Fusion_Block(input_2Dfeature_channel, input_channel//POS_NUM, feature_channel,
                                                  kernel_size_grav, scale_num, dataset_name)
            setattr(self, 'IMU_fusion_blocks%i' % i, IMU_fusion_block)
            self.IMU_fusion_blocks.append(IMU_fusion_block)
        
        self.perform_local_temporal_fusion = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.PReLU(),
            )
        
        self.brush_conv_0 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.PReLU(),
            )
        
        self.brush_conv_1 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.brush_conv_2 = nn.Sequential(
            nn.Conv2d(feature_channel_out, feature_channel_out, (1,3), 1, (0,3//2)),
            nn.BatchNorm2d(feature_channel_out),
            nn.PReLU(),
            )
        
        self.linear_high1 = nn.Linear(3*feature_channel, self.feature_channel_out)
        
        self.DWT_2D       = DWT_2D(in_channels = feature_channel, INFERENCE_DEVICE=INFERENCE_DEVICE)
        self.DWT_1D       = DWT_1D(in_channels = feature_channel_out, INFERENCE_DEVICE=INFERENCE_DEVICE)
        
        if input_channel//POS_NUM  == 12:
            reduced_channel = 6
        else:
            reduced_channel = 3
        
        gragh                   = self.create_perstamp_gragh((9-reduced_channel)//3*POS_NUM, INFERENCE_DEVICE)
        
        self.create_large_gragh(gragh, train_size, val_size, test_size, data_length, BATCH_SIZE, test_split)
        
        self.HeteGNNsubnet      = HeteGNN(feature_channel, feature_channel, feature_channel_out, 0.2, POS_NUM)
        
        self.graph_ave_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.graph_max_pooling = nn.AdaptiveMaxPool1d(1)
        
        self.gumbel_block0 = gumble_block_2D(feature_channel*2, 2, (input_channel//POS_NUM-reduced_channel)*scale_num, data_length//2)
        
        self.gumbel_block1 = gumble_block_2D(feature_channel*2, 2, (input_channel//POS_NUM-reduced_channel)*scale_num, data_length//4)
        
        self.gumbel_block2 = gumble_block_1D(feature_channel_out*2, 2, data_length)
        
        self.position_encode = PositionalEncoding(feature_channel_out, drop_rate, data_length//4)
        
        self.perform_global_temporal_fusion_1 = Transformer_1(feature_channel_out, multiheads, drop_rate)
        
        self.perform_global_temporal_fusion_2 = Transformer_2(feature_channel_out, multiheads, drop_rate)
        
        self.global_ave_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.linear             = nn.Linear(feature_channel_out, num_class)
        
    def create_perstamp_gragh(self, node_num, INFERENCE_DEVICE):
        node_set         = np.arange(node_num).tolist()
        g_ego            = []
        g_nb             = []
        for node_id in range(node_num):
            nb_node_set  = node_set.copy()
            nb_node_set.remove(node_id)
            ego_nodes    = [node_id] * len(nb_node_set)
            g_ego        = g_ego     + ego_nodes
            g_nb         = g_nb      + nb_node_set
        # cal indegree
        gragh = dgl.graph((g_ego, g_nb))
        deg   = gragh.in_degrees().float().clamp(min=1)
        if INFERENCE_DEVICE == 'TEST_CUDA':
            gragh = gragh.to('cuda')
            deg   = deg.cuda()
        norm  = torch.pow(deg, -0.5)
        gragh.ndata['d'] = norm
        return gragh

    def create_large_gragh(self, gragh, train_size, val_size, test_size, data_length, BATCH_SIZE, test_split):
        self.train_size         = train_size
        self.val_size           = val_size
        self.test_size          = test_size
        self.batch_g            = dgl.batch([gragh] * (BATCH_SIZE*(data_length//4)))
        if train_size%BATCH_SIZE != 0:
            self.batch_last_g   = dgl.batch([gragh] * (train_size%BATCH_SIZE*(data_length//4)))
        else:
            self.batch_last_g   = []
        self.tr_g               = dgl.batch([gragh] * (train_size//test_split*(data_length//4)))
        if train_size%(train_size//test_split) != 0:
            self.tr_last_g      = dgl.batch([gragh] * (train_size%(train_size//test_split)*(data_length//4)))
        else:
            self.tr_last_g      = []
        self.val_g              = dgl.batch([gragh] * (val_size//test_split*(data_length//4)))
        if val_size%(val_size//test_split) != 0:
            self.val_last_g     = dgl.batch([gragh] * (val_size%(val_size//test_split)*(data_length//4)))
        else:
            self.val_last_g     = []
        self.test_g             = dgl.batch([gragh] * (test_size//test_split*(data_length//4)))
        if test_size%(test_size//test_split) != 0:
            self.test_last_g    = dgl.batch([gragh] * (test_size%(test_size//test_split)*(data_length//4)))
        else:
            self.test_last_g    = []
        self.flops_g            = dgl.batch([gragh] * (data_length//4))

    def generate_batch_gragh(self, batch_size, BATCH_SIZE, test_split):
        # batch_size      = batch_size//2
        if batch_size   == BATCH_SIZE:
            batch_gragh = self.batch_g
        elif batch_size == self.train_size%BATCH_SIZE:
            batch_gragh = self.batch_last_g
        elif batch_size == self.train_size//test_split:
            batch_gragh = self.tr_g
        elif batch_size == self.train_size%(self.train_size//test_split):
            batch_gragh = self.tr_last_g
        elif batch_size == self.val_size//test_split:
            batch_gragh = self.val_g
        elif batch_size == self.val_size%(self.val_size//test_split):
            batch_gragh = self.val_last_g
        elif batch_size == self.test_size//test_split:
            batch_gragh = self.test_g
        elif batch_size == self.test_size%(self.test_size//test_split):
            batch_gragh = self.test_last_g
        elif batch_size == 1:
            batch_gragh = self.flops_g
        # print(batch_size)
        return batch_gragh
    
    def perform_IMU_fusion(self, x_input, IMU_num):
        
        for i in range(IMU_num):
            x_cur_IMU, cur_sensor_attn   = self.IMU_fusion_blocks[i](x_input[:,:,i*9:(i+1)*9,:])
            if i == 0:
                x         = x_cur_IMU
                IMU_attns = cur_sensor_attn
            else:
                x         = torch.cat((x, x_cur_IMU), 2)
                IMU_attns = torch.cat((IMU_attns, cur_sensor_attn), 2)
        return x, IMU_attns
    
    def perform_modality_node_fusion(self, x_low1, x_high1, batch_size, data_length, IMU_num):
        
        # Convert data size to meet HeteGNN's input requrements.
        x_low1  = x_low1.permute(0,3,2,1)
        x_low1  = x_low1.reshape(batch_size, data_length//4, IMU_num, -1, 3 * self.feature_channel)
        x_high1 = x_high1.permute(0,3,2,1)
        x_high1 = x_high1.reshape(batch_size, data_length//4, IMU_num, -1, 3 * self.feature_channel)
        
        # Transform x_high1 to be a residual term.
        x_high1 = self.linear_high1(x_high1).reshape(batch_size*(data_length//4), -1, self.feature_channel_out)
        
        # Generate heterogeneous graph.
        batch_gragh = self.generate_batch_gragh(batch_size, self.BATCH_SIZE, self.test_split)
        # Adaptive filtering in graph Fourier domain.
        x_low1, Graph_attns = self.HeteGNNsubnet(batch_gragh, x_low1)
        
        x_low1 = x_low1.reshape(batch_size*(data_length//4), -1, self.feature_channel_out)
        
        # Perform graph pooling.
        x = torch.cat([x_low1, x_high1], dim=0)
        x = x.reshape(2*batch_size*(data_length//4), -1, self.feature_channel_out)
        x = x.permute(0,2,1)
        x = self.graph_ave_pooling(x).squeeze(-1)
        x = x.reshape(2*batch_size, data_length//4, -1)
        
        return x, Graph_attns
    
    def perform_frequency_selection_0(self, x, test_flag):
        
        x_low0, x_high0 = self.DWT_2D(x)
        x               = torch.cat([x_low0, x_high0], dim=2)
        
        x               = self.brush_conv_0(x) #+ x
        
        x_low0          = x[:,:,0:x.shape[2]//2,:]
        x_high0         = x[:,:,x.shape[2]//2:x.shape[2],:]
        x_low0, x_high0, ch_mask_0 = self.gumbel_block0(x_low0, x_high0, test_flag)
        
        return x_low0, x_high0, ch_mask_0
    
    def perform_frequency_selection_1(self, x, test_flag):
        
        x_low1, x_high1 = self.DWT_2D(x)
        x               = torch.cat([x_low1, x_high1], dim=2)
        
        x               = self.brush_conv_1(x) #+ x
        
        x_low1          = x[:,:,0:x.shape[2]//2,:]
        x_high1         = x[:,:,x.shape[2]//2:x.shape[2],:]
        x_low1, x_high1, ch_mask_1 = self.gumbel_block1(x_low1, x_high1, test_flag)
        
        return x_low1, x_high1, ch_mask_1
    
    def perform_frequency_selection_2(self, x, test_flag):
        
        x_low2, x_high2 = self.DWT_1D(x)
        x               = torch.cat([x_low2.unsqueeze(2), x_high2.unsqueeze(2)], dim=2)
        
        x               = self.brush_conv_2(x) #+ x

        x_low2          = x[:,:,0,:]
        x_high2         = x[:,:,1,:]
        x_low2, x_high2, ch_mask_2 = self.gumbel_block2(x_low2, x_high2, test_flag)
        
        return x_low2.permute(0,2,1), x_high2.permute(0,2,1), ch_mask_2
    
    def forward(self, x, test_flag=False):
        
        # flops
        if len(x.shape) == 3:
            x           = x.unsqueeze(0)
        # flops
        
        batch_size      = x.shape[0]
        feature_channel = x.shape[1]
        input_channel   = x.shape[2]
        data_length     = x.shape[-1]
        IMU_num         = self.POS_NUM
        x_input         = x
        
        ### IMU fusion block
        x, IMU_attns               = self.perform_IMU_fusion(x_input, IMU_num)
        
        ### Local temporal fusion block ###
        x_low0, x_high0, ch_mask_0 = self.perform_frequency_selection_0(x, test_flag)
        x                          = self.perform_local_temporal_fusion(x_low0) + x_high0
        
        ### Golabal temporal fusion block 1 ###
        x_low1, x_high1, ch_mask_1 = self.perform_frequency_selection_1(x, test_flag)
        ### Modality node fusion block ###
        x, Graph_attns             = self.perform_modality_node_fusion(x_low1, x_high1, batch_size, data_length, IMU_num)
        # perform global temporal fusion
        x_high1                    = x[x.shape[0]//2:x.shape[0],:,:]
        x_low1                     = self.position_encode(x[0:x.shape[0]//2,:,:].permute(0,2,1))
        x                          = self.perform_global_temporal_fusion_1(x_low1.permute(0,2,1), x_high1)
        
        ### Golabal temporal fusion block 2 ###
        x_low2, x_high2, ch_mask_2 = self.perform_frequency_selection_2(x, test_flag)
        x                          = self.perform_global_temporal_fusion_2(x_low2, x_high2)
        
        x                          = self.global_ave_pooling(x).squeeze(-1)
        
        output                     = self.linear(x)
        
        return output, [IMU_attns, Graph_attns, torch.cat((ch_mask_0,ch_mask_1,ch_mask_2),1)]
    
class MixUpLoss(nn.Module):

    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output, target[:, 0].long()), self.crit(output, target[:, 1].long())
            d = loss1 * target[:, 2] + loss2 * (1 - target[:, 2])
        else:
            d = self.crit(output, target)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d

    def get_old(self):
        if hasattr(self, 'old_crit'):
            return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit

def mixup_data(x, y, alpha=0.4):

    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha, batch_size)
    # t = max(t, 1-t)
    lam = np.concatenate([lam[:, None], 1 - lam[:, None]], 1).max(1)
    # tensor and cuda version of lam
    lam = x.new(lam)

    shuffle = torch.randperm(batch_size).cuda()

    x1, y1 = x[shuffle], y[shuffle]
    # out_shape = [bs, 1, 1]
    out_shape = [lam.size(0)] + [1 for _ in range(len(x1.shape) - 1)]

    # [bs, temporal, sensor]
    mixed_x = (x * lam.view(out_shape) + x1 * (1 - lam).view(out_shape))
    # [bs, 3]
    y_a_y_b_lam = torch.cat([y[:, None].float(), y1[:, None].float(), lam[:, None].float()], 1)

    return mixed_x, y_a_y_b_lam

def train_op(network, EPOCH, BATCH_SIZE, LR,
             train_x, train_y, val_x, val_y, X_test, y_test,
             output_directory_models, log_training_duration, test_split):
    
    # prepare training_data
    if train_x.shape[0] % BATCH_SIZE == 1:
        drop_last_flag = True
    else:
        drop_last_flag = False
    torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
    train_loader = Data.DataLoader(dataset = torch_dataset,
                                    batch_size = BATCH_SIZE,
                                    shuffle = True,
                                    drop_last = drop_last_flag
                                    )
    
    # init lr&train&test loss&acc log
    lr_results = []
    
    loss_train_results = []
    accuracy_train_results = []
    
    loss_validation_results = []
    accuracy_validation_results = []
    macro_f1_val_results        = []
    
    loss_test_results = []
    accuracy_test_results = []
    macro_f1_test_results       = []
    
    parameters = network.parameters()
    optimizer  = torch.optim.Adam(parameters,lr = LR)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                                            patience=5,
                                                            min_lr=LR/10, verbose=True)
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_function_nomixup = LabelSmoothingCrossEntropy()
    
    # save init model
    output_directory_init = os.path.join(output_directory_models, 'init_model.pkl')
    torch.save(network.state_dict(), output_directory_init)   # only save the init parameters
    
    training_duration_logs = []
    
    # super param
    mixup = True
    alpha = 0.8
    #############
    
    start_time = time.time()
    for epoch in range (EPOCH):
        
        epoch_tau = epoch+1
        tau = max(1 - (epoch_tau - 1) / 50, 0.5)
        for m in network.modules():
            if hasattr(m, '_update_tau'):
                m._update_tau(tau)
                # print(a)
        
        for step, (x,y) in enumerate(train_loader):
            
            batch_x = x.cuda()
            batch_y = y.cuda()
            
            if mixup == True:
                batch_x, batch_y_mixup = mixup_data(batch_x, batch_y, alpha)
            
            logits, _ = network(batch_x)
            
            # cal the sum of pre loss per batch
            if mixup == True:
                loss_function    = MixUpLoss(criterion)
                loss             = loss_function(logits, batch_y_mixup)
            else:
                loss             = loss_function_nomixup(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if mixup == True:
                loss_function = loss_function.get_old()
        
        # test per epoch
        network.eval()
        test_flag = True
        # loss_train:loss of training set; accuracy_train:pre acc of training set
        loss_train, accuracy_train, _ = get_test_loss_acc_dynamic(network, loss_function, train_x, train_y, test_split, test_flag)
        loss_validation, accuracy_validation, macro_f1_val = get_test_loss_acc_dynamic(network, loss_function, val_x, val_y, test_split, test_flag)
        loss_test, accuracy_test, macro_f1_test = get_test_loss_acc_dynamic(network, loss_function, X_test, y_test, test_split, test_flag)
        test_flag = False
        network.train()
        
        # update lr
        scheduler.step(accuracy_validation)
        lr = optimizer.param_groups[0]['lr']
        
        # log lr&train&validation loss&acc per epoch
        lr_results.append(lr)
        loss_train_results.append(loss_train)    
        accuracy_train_results.append(accuracy_train)
        
        loss_validation_results.append(loss_validation)    
        accuracy_validation_results.append(accuracy_validation)
        macro_f1_val_results.append(macro_f1_val)
        
        loss_test_results.append(loss_test)    
        accuracy_test_results.append(accuracy_test)
        macro_f1_test_results.append(macro_f1_test)
        
        # print training process
        if (epoch+1) % 1 == 0:
            print('Epoch:', (epoch+1), '|lr:', lr,
                  '| train_loss:', loss_train, 
                  '| train_acc:', accuracy_train, 
                  '| validation_loss:', loss_validation, 
                  '| validation_acc:', accuracy_validation)
        
        save_models(network, output_directory_models, 
                    loss_train, loss_train_results, 
                    accuracy_validation, accuracy_validation_results,
                    start_time, training_duration_logs)
    
    # log training time 
    per_training_duration = time.time() - start_time
    log_training_duration.append(per_training_duration)
    
    # save last_model
    output_directory_last = os.path.join(output_directory_models, 'last_model.pkl')
    torch.save(network.state_dict(), output_directory_last)   # save only the init parameters
    
    # log history
    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                          loss_validation_results, accuracy_validation_results,
                          loss_test_results, accuracy_test_results,
                          output_directory_models)
    
    plot_learning_history(EPOCH, history, output_directory_models)
    
    return(history, per_training_duration, log_training_duration)