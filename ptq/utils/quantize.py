
import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class UniformQuantize(InPlaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=8, min_value=None, max_value=None, inplace=False, symmetric=False, num_chunks=None):
        num_chunks = input.shape[0] if not num_chunks else num_chunks
        if not min_value or not max_value:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)
        
        if not min_value:
            min_value = y.min(-1)[0].mean(-1)

        if not max_value:
            max_value = y.max(-1)[0].mean(-1)

        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if symmetirc:
            qmin = -2. ** (num_bits - 1)
            qmax = 2 ** (num_bits - 1) - 1
            max_value = abs(max_value)
            min_value = abs(min_value)
            if max_vaule < min_value:
                max_value = min_value
            scale = max_value / qmax

            min_value = 0.

        else:
            qmin = 0.
            qmax = 2. ** num_bits - 1.

            scale = (max_value - min_value) / (qmax - qmin)
            
        scale = max(scale, 1e-8)

        # round(clip((x - z)/s)) * s + z
        output.add_(-min_value).div_(scale)

        output.clamp_(qmin, qmax).round_()

        output.mul_(scale).add_(min_value)

        return output

def quantize(x, num_bits=8, min_value=None, max_value=None, inplace=False, symmetric=False, num_chunks=None):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, inplace, symmetric, num_chunks)

class QuantMeasure(nn.Module):

    def __init__(self, update_stat=False, num_bits=8, momentum=0.1):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits
        self.update_stat = update_stat

    def forward(self, input):
        if self.update_stat:
            self.running_max = max(self.running_max, input.detach().view(input.size(0), -1).max(-1)[0].mean())
            self.running_min = min(self.running_min, input.detach().view(input.size(0), -1).min(-1)[0].mean())

        if self.training:
            min_value = input.detach().view(input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(input.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(1 - self.momentum).add_(min_value * (self.momentum))
            self.running_max.mul_(1 - self.momentum).add_(max_value * (self.momentum))

        else:
            min_value = self.running_min
            max_value = self.running_max

        return quantize(input, self.num_bits, min_value = float(max_value), num_chunks=16)

    def set_update_stat(self, update_stat):
        self.update_stat = update_stat

class QConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_act=8, num_bits+bias=16, momentum=0.1):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias, num_bits, num_bits_act, num_bits+bias, momentum)

        self.num_bits = num_bits
        self.num_bits_bias = num_bits_bias
        self.quant = QuantMeasure(num_bits=numbits_act, momentum = momentum)

    def set_scale(self, scale=None, scale_prev=None):
        if scale:
            self.register_parameter("scale", nn.Parameter(scale.view(-1, 1, 1, 1)))

        if scale_prev:
            self.scale_prev = scale_prev

    def merge_scale_to_weight(self):
        if hasattr(self, 'scale_prev') and self.scale_prev:
            weight = self.merge_scale_prev(self.weight.detach(), self.scale_prev)
            self.weight.data.copu_(weight)
            self.scale_prev = None

        if hasattr(self, 'scale') and self.scale:
            weight, bias = self.merge_scale(self.weight.detach(), self.bias.detach() if self.bias else self.bias, self.scale)
            self.weight.data.copy_(weight)
            if self.bias:
                self.bias.data.copy_(bias)
            self.scale = None

    def merge_scale_prev(self, weight, scale_prev):
        sweight = weight.clone()
        step = weight.shape[0] // self.groups
        step_s = weight.shape[1]

        scale_prev = scale_prev[:, 0, 0, 0].view(1, -1, 1, 1)
        for g in range(self.groups):
            sweight[g*step:(g+1)*step] =weight[g*step: (g+1)*step] / scale_prev[:, g*step_s:(g+1)*step_s]

        return sweight

    def merge_scale(self, weight, bias, scale):
        weight = weight * scale
        if bias:
            bias = bias * scale.view(-1)

        return weight, bias

    def forward(self, input):
        input = self.quant(input)
        sweight = self.weight.clone()
        sbias = self.bias
        if hasattr(self, 'scale_prev') and self.scale_prev:
            scale = self.scale_prev
            sweight = self.merge_scale_prev(self.weight, scale)
        else:
            sweight = self.weight

        if hasattr(self, 'scale') and self.scale:
            scale = self.scale
            sweight, sbias = self.merge_scale(sweight, sbias, scale)

        qweight = quantize(sweight, num_bits=self.num_bits,
                          min_value=float(sweight.min()),
                          max_value=float(sweight.max()))
        if sbias:
            qbias = quantize(sbias, num_bits=self.num_bits_bias)
        else:
            qbias = None

        output = F.conv2d(input, qweight, qbias, self.stride,
                         self.padding, self.dilation, self.groups)

        return output

class QuantConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_act=8, num_bits_bias=16, momentum=0.1):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_bias = num_bits_bias 
        self.quant = QuantMeasure(num_bits=num_bits_act, momentum=momentum)
















