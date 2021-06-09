# YOLOv5 pruned common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.cuda import amp
from models.common import Conv


class BottleneckPruned(nn.Module):
    # Pruned bottleneck
    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1):  # ch_in, ch_out, shortcut, groups, expansion
        super(BottleneckPruned, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out, cv2out, 3, 1, g=g)
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3Pruned(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, cv1in, cv1out, cv2out, cv3out, bottle_args, n=1, shortcut=True, g=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3Pruned, self).__init__()
        cv3in = bottle_args[-1][-1]
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)
        self.cv3 = Conv(cv3in+cv2out, cv3out, 1)
        self.m = nn.Sequential(*[BottleneckPruned(*bottle_args[k], shortcut, g) for k in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3PrunedNoRes(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, cv1in, cv1out, cv2out, cv3out):  # ch_in, ch_out
        super(C3PrunedNoRes, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)
        self.cv3 = Conv(cv1out+cv2out, cv3out, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.cv1(x), self.cv2(x)), dim=1))

class SPPPruned(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, cv1in, cv1out, cv2out, k=(5, 9, 13)):
        super(SPPPruned, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out * (len(k) + 1), cv2out, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


