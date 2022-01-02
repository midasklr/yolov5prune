# -*- coding: utf-8 -*-
# @Time : 2021/5/24 下午4:36 
# @Author : midaskong 
# @File : prune_utils.py 
# @Description:

import torch
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


def gather_bn_weights(module_list):
    prune_idx = list(range(len(module_list)))
    size_list = [idx.weight.data.shape[0] for idx in module_list.values()]
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for i, idx in enumerate(module_list.values()):
        size = size_list[i]
        bn_weights[index:(index + size)] = idx.weight.data.abs().clone()
        index += size
    return bn_weights

def gather_conv_weights(module_list):
    prune_idx = list(range(len(module_list)))
    size_list = [idx.weight.data.shape[0] for idx in module_list.values()]

    conv_weights = torch.zeros(sum(size_list))
    index = 0
    for i, idx in enumerate(module_list.values()):
        size = size_list[i]
        conv_weights[index:(index + size)] = idx.weight.data.abs().sum(dim=1).sum(dim=1).sum(dim=1).clone()
        index += size
    return conv_weights


def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask


def obtain_conv_mask(conv_module, thre):
    thre = thre.cuda()
    mask = conv_module.weight.data.abs().sum(dim=1).sum(dim=1).sum(dim=1).ge(thre).float()
    return mask

def uodate_pruned_yolov5_cfg(model, maskbndict):
    # save pruned yolov5 model in yaml format:
    # model:
    #   model to be pruned
    # maskbndict:
    #   key     : module name
    #   value   : bn layer mask index
    return