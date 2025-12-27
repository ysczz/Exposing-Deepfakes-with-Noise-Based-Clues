
"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch
from torch import tensor
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import os 
import config 
import numpy as np
from torch.nn.parameter import Parameter
from net.srm import setup_srm_layer

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

class SegModule(nn.Module):
    def __init__(self, dim):
        super(SegModule, self).__init__()
        self.dim = dim
        self.kernel_size = 3
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        factor = 4
        final_factor=2
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor,dim//final_factor, kernel_size=1),
            nn.BatchNorm2d(dim//final_factor),
            nn.ReLU(inplace=True),
        )

        self.h_size = 10 
        self.MemoryFeature = Parameter(torch.rand(dim, self.h_size, self.h_size))

        self.convert = nn.Sequential(
            nn.Conv2d( self.h_size * self.h_size , dim//final_factor, 1, bias=False),
            nn.BatchNorm2d(dim//final_factor),
            nn.ReLU(inplace=True),
        )
        
        self.seg = nn.Conv2d(in_channels=dim, out_channels=1,
                         kernel_size=3, stride=1, padding=1)
        self.s = nn.Sigmoid()

        
        
    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        w = self.embed(qk)  # b, dim//2 ,h ,w

        batch_size = x.size(0)
        theta_x = x.view(batch_size,  self.dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        memory = self.MemoryFeature.expand(x.size())
        phi_x = memory.view(batch_size, self.dim, -1)
        y = torch.matmul(theta_x, phi_x)
        y =  y.view(batch_size, -1, self.h_size, self.h_size)
        y = self.convert(y)
        y = torch.cat((w, y),1)
        relation = y
        y = self.seg(y)
        y = self.s(y)
        return y, relation


class NoiseExtract(nn.Module):
    def __init__(self,inchannel):
            super(NoiseExtract, self).__init__()
            self.inchannel = inchannel
            self.srm_conv = setup_srm_layer(input_channels=inchannel)
            # self.sconv = SeparableConv2d(channel,channel,1,stride=1,bias=False)
            self.conv = nn.Conv2d(3,inchannel,1,stride=1, bias=False)
            # self.bn = nn.BatchNorm2d(channel)
            # self.s = nn.Sigmoid()
            # self.norm = norm
            # self.convert = convert

    def forward(self, input):
        # feature = self.filter_factory('avg', input)
        # if self.convert==True:
        #     s_feature = self.sconv(feature)
        #     n_feature = self.conv(feature)
        #     feature = s_feature + n_feature
        # feature = self.bn(feature)
        # feature = self.s(feature)
        feature = self.srm_conv(input)
        if self.inchannel != 3:
            feature = self.conv(feature)
        return feature

    def filter_factory(self, name, feature):
        if name=='avg':
            avg_feature =  F.avg_pool2d(feature, kernel_size=3,stride=1,padding=1)
     
            feature = feature - avg_feature
            return feature
        if name=='midium':
            return null



class Xception_Base(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=2):
        super(Xception_Base, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)
        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)
        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(4096, num_classes)
        self.dp = nn.Dropout(p=0.2)

        # stream two 
        self.noise_extract_layer1 = NoiseExtract(32)
        self.noise_extract_layer2 = NoiseExtract(64)

    

    def srm_features(self, input, add1, add2):
        x = self.conv1(input)
        x = x + self.noise_extract_layer1(add1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + self.noise_extract_layer2(add2)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        seg_srm_feature =x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x, seg_srm_feature

    def features(self, input):
        x = self.conv1(input)
        srm_add1 = x
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        srm_add2 = x
        x = self.bn2(x)
        x = self.relu(x)
      
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        seg_feature =x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)

        return  x, srm_add1, srm_add2, seg_feature

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        fea = x.view(x.size(0), -1)
        fea = self.dp(fea)
        x = self.fc(fea)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

class SRM(nn.Module):
    def __init__(self):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(SRM, self).__init__()
        self.RGB = Xception_Base()
        self.SRM = Xception_Base()
        state_dict = get_xception_dict()
        self.RGB.load_state_dict(state_dict, False)
        self.SRM.load_state_dict(state_dict, False)
        self.noise_extract_input = NoiseExtract(3)
        self.fc = nn.Linear(6144, 2)
        self.dp = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(inplace=True)
        self.relation_module = SegModule(2048)
       
        
    def features(self, rgb, srm):
        rgb_x, srm_add1, srm_add2, seg_feature = self.RGB.features(rgb)
        srm_x, seg_srm_feature = self.SRM.srm_features(srm, srm_add1, srm_add2)
        feature = torch.cat((seg_feature,seg_srm_feature),1)
        seg, relation = self.relation_module(feature)
        x = torch.cat((rgb_x,srm_x),1)
        x = torch.cat((x,relation),1)
        return x, seg
       

    
    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        fea = x.view(x.size(0), -1)
        fea = self.dp(fea)
        x = self.fc(fea)
        return x

    def forward(self, rgb):
        srm=  self.noise_extract_input(rgb)
        x,seg = self.features(rgb, srm)
        x = self.logits(x)
        return x, seg


def get_xception_dict(pretrained_path='net/xception.pth'):
    print('load static', pretrained_path)
    state_dict = torch.load(os.path.join(config.ROOT_PATH, pretrained_path))
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict
   

