# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/9/25 12:50
# @Author : zhoujing
# @File : msffa.py
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# from model.conv.CondConv import CondConv

class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained_path=None):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=not pretrained_path)
            self.final_out_channels = 256
            self.low_level_inplanes = 64
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=not pretrained_path)
            self.final_out_channels = 256
            self.low_level_inplanes = 64
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=not pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=not pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=not pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        if pretrained_path:
            backbone.load_state_dict(torch.load(pretrained_path))


        self.early_extractor = nn.Sequential(*list(backbone.children())[:5])
        self.later_extractor = nn.Sequential(*list(backbone.children())[5:7])

        conv4_block1 = self.later_extractor[-1][0]

        self.ada = self.later_extractor[0]

        # conv4_block1.conv1.stride = (1, 1)
        # conv4_block1.conv2.stride = (1, 1)
        # conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.early_extractor(x)
        t = self.ada(x)
        out = self.later_extractor(x)
        return out,x

class MyConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(MyConv, self).__init__()
        # self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,stride=1, padding=padding, dilation=dilation, bias=False)
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        # self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,stride=1, padding=padding, dilation=dilation, bias=False)
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes=2048, output_stride=16):
        super(ASPP, self).__init__()
        if output_stride == 16:
            # dilations = [1, 6, 12, 18]
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # return self.dropout(x),x1,x2,x3,x4
        return self.dropout(x),x1,x2,x3,x4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPP2(nn.Module):
    def __init__(self, inplanes=2048, output_stride=16):
        super(ASPP2, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
            # dilations = [1, 12, 24, 36]
            # dilations = [1, 3, 6, 12]
            # dilations = [1, 3, 6, 9]
            # dilations = [1, 24, 48, 72]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.len = 64
        self.aspp1 = _ASPPModule(inplanes, self.len, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, self.len, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, self.len, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, self.len, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, self.len, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.len),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(320, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes=256):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x1 = F.interpolate(x[:,:128], size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[:,128:], size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1,x2],dim=1)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class mutilattention(nn.Module):
    def __init__(self, num_classes, low_level_inplanes=256):
        super(Decoder3, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, low_level_inplanes, 1, bias=False)

class Decoder3(nn.Module):
    def __init__(self, num_classes, low_level_inplanes=256):
        super(Decoder3, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, low_level_inplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(low_level_inplanes)
        self.relu = nn.ReLU()

        self.conv_a1 = nn.Conv2d(128, 64, 1, bias=False)
        self.conv_a2 = nn.Conv2d(128, 64, 1, bias=False)

        self.conv_b1 = nn.Conv2d(128, 64, 1, bias=False)
        self.conv_b2 = nn.Conv2d(128, 64, 1, bias=False)

        # self.mhsa = MultiHeadSelfAttention(64)

        self.last_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True))

        self.last_conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(64, num_classes, kernel_size=1, stride=1))

        self._init_weight()


    def forward(self, x, low_level_feat, x_oro= None):

        xsplit = torch.split(x,dim=1,split_size_or_sections=128)

        xa1 = F.interpolate(self.conv_a1(xsplit[0]),size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        xa2 = self.conv_a2(F.interpolate(xsplit[0], size=low_level_feat.size()[2:], mode='bilinear', align_corners=True))

        xa = xa1 - xa2
        xalow = self.relu(self.conv1(low_level_feat) + self.conv1(xa))

        xb1 = F.interpolate(self.conv_b1(xsplit[1]), size=low_level_feat.size()[2:], mode='bilinear',align_corners=True)
        xb2 = self.conv_b2(F.interpolate(xsplit[1], size=low_level_feat.size()[2:], mode='bilinear', align_corners=True))
        xb = xb1 - xb2
        xblow = self.relu(self.conv1(low_level_feat) + self.conv1(xb))

        low_level_feat = xalow + xblow


        # x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)

        # x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(low_level_feat)

        x = F.interpolate(x, size=x_oro.size()[2:], mode='bilinear', align_corners=True)


        x = (x + x_oro)/2
        x = self.last_conv2(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class _ASPPModule22(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule22, self).__init__()
        # self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,stride=1, padding=padding, dilation=dilation, bias=False)

        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.Softmax2d = nn.Softmax2d()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        return self.Softmax2d(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP22(nn.Module):
    def __init__(self, inplanes=2048, output_stride=16):
        super(ASPP22, self).__init__()
        if output_stride == 16:
            # dilations = [1, 6, 12, 18]
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule22(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule22(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule22(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule22(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = x *x1
        x2 = self.aspp2(x)
        x2 = x * x2
        x3 = self.aspp3(x)
        x3 = x* x3
        x4 = self.aspp4(x)
        x4 = x * x4
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # return self.dropout(x),x1,x2,x3,x4
        return self.dropout(x),x1,x2,x3,x4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()

class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)
        return x

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.pe = PositionalEncodingPermute2D(channel)

    def forward(self, x):
        b, c, h, w = x.size()
        # pe = self.positional_encoding_2d(c, h, w)
        pe = self.pe(x)
        x = x + pe
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) /
                         math.sqrt(c))  #[b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x

class Decoder4(nn.Module):
    def __init__(self, num_classes, low_level_inplanes=256):
        super(Decoder4, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, stride=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EnDecoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes=256):
        super(EnDecoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes=None):

        super().__init__()
        self.num_classes = num_classes
        self.input_channels = 3
        self.deep_supervision = None

        self.backbone = ResNet('resnet34', None)
        self.aspp = ASPP22(inplanes=self.backbone.final_out_channels)  ##final_out_channels == 1024

        self.conv1 = nn.Sequential(nn.Conv2d(3,16,kernel_size=3,padding='same'),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16,64,kernel_size=1,padding='same'))

        self.decoder2 = Decoder3(num_classes, self.backbone.low_level_inplanes)


        self.decoder3 = Decoder3(num_classes, self.backbone.low_level_inplanes)

        self.atten2scale = nn.Sequential(nn.Conv2d(64,2,kernel_size=3,padding='same',bias=False),
                                         nn.Conv2d(2,2,kernel_size=1,bias=False))

        self.nnm = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU())
        self.f100maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)


    def forward(self, imgs100, imgs50=None, imgs75=None, labels=None, mode='infer', **kwargs):

        x100, low_level_feat100 = self.backbone(imgs100)
        x01 = self.conv1(imgs100)
        x_oro = x01
        x100,x1,x2,x3,x4 = self.aspp(x100)
        x100 = self.decoder2(x100, low_level_feat100,x_oro)

        outputs100 = F.interpolate(x100, size=imgs100.size()[2:], mode='bilinear', align_corners=True)


        if self.training:
            return outputs100
        else:
            return outputs100

if __name__ == '__main__':
    model = DeepLabv3Plus(num_classes=19)
