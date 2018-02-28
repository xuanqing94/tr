#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
from models.net import VGG, Test
from math import sqrt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, required=True)
    parser.add_argument('--modelIn1', type=str, required=True)
    parser.add_argument('--modelIn2', type=str, required=True)

    opt = parser.parse_args()
    net1 = VGG("VGG16")
    net2 = VGG("VGG16")
    net1 = nn.DataParallel(net1, device_ids=range(opt.ngpu))
    net2 = nn.DataParallel(net2, device_ids=range(opt.ngpu))
    net1.load_state_dict(torch.load(opt.modelIn1))
    net2.load_state_dict(torch.load(opt.modelIn2))
    loss_f = nn.CrossEntropyLoss()
    net1.cuda()
    net2.cuda()
    dist = 0.0
    net2_state = net2.state_dict()
    for k, v in net1.state_dict().items():
        diff = v - net2_state[k]
        dist += torch.sum(diff * diff)
    print("model1: {}, model2: {}, L2-dist: {}".format(opt.modelIn1, opt.modelIn2, sqrt(dist)))
