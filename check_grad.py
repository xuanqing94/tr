#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dst
import torchvision.transforms as tfs
from models.net import VGG, Test
from torch.utils.data import DataLoader
import time

def Hv_exact(net, loss_f, input, output, v):
    loss  = loss_f(net(input), output)
    grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=True)
    Hv = torch.autograd.grad(grad_params, net.parameters(), grad_outputs=v)
    return Hv

def Hv_exact2(net, loss_f, input, output, v):
    loss  = loss_f(net(input), output)
    grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=True)
    inner = 0
    for k1, k2 in zip(grad_params, v):
        inner += torch.sum(k1 * k2)
    Hv = torch.autograd.grad(inner, net.parameters())
    return Hv

def Hv_approx(net, loss_f, input, output, v, scale=100000.0):
    loss  = loss_f(net(input), output)
    grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=False)
    for p, vi in zip(net.parameters(), v):
        p.data += vi.data / scale
    loss  = loss_f(net(input), output)
    grad_params_ = torch.autograd.grad(loss, net.parameters(), create_graph=False)
    Hv = []
    for g1, g2 in zip(grad_params, grad_params_):
        Hv.append((g2 - g1) * scale)
    for p, vi in zip(net.parameters(), v):
        p.data -= vi.data / scale
    return Hv

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init2(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 1)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 1)
        m.bias.data.fill_(0)
def test_grad():
    from models.activate import SReLU
    x = Variable(torch.FloatTensor(16, 3 * 32 * 32).normal_(), requires_grad=True)
    label = Variable(torch.FloatTensor(32, 10).normal_())
    linear = nn.Linear(3 * 32 * 32, 10, bias=False)

    relu = SReLU()
    v = Variable(torch.FloatTensor(3 * 32 * 32 * 10).normal_())
    y = relu(linear(x))
    err = label - y
    y = torch.sum(err * err)
    g = torch.autograd.grad(y, linear.parameters(), create_graph=True, only_inputs=True)
    g_v = torch.cat([e.view(-1) for e in g])
    inner = torch.sum(g_v * v)
    Hv1 = torch.autograd.grad(g_v, linear.parameters(), grad_outputs=v, only_inputs=True)
    Hv2 = torch.autograd.grad(inner, linear.parameters())
    print(Hv1, Hv2)

if __name__ == "__main__":
    from models.activate import SReLU
    #linear = nn.Linear(3 * 32 * 32, 10)
    #relu = nn.ReLU()
    #linear2 = nn.Linear(10, 10)
    #net = nn.Sequential(linear, relu, linear2)
    net = VGG("VGG16")
    net = nn.DataParallel(net, device_ids=range(4))
    net.apply(weights_init)
    loss_f = nn.CrossEntropyLoss()
    net.cuda()
    loss_f.cuda()
    transform_train = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data = dst.CIFAR10("~/data/cifar10-py", download=True, train=True, transform=transform_train)
    data_test = dst.CIFAR10("~/data/cifar10-py", download=True, train=False, transform=transform_test)
    assert data, data_test
    dataloader = DataLoader(data, batch_size=16, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(data_test, batch_size=16, shuffle=False, num_workers=2)
    for x, y in dataloader:
        #x = x.view(x.size(0), -1)
        input = Variable(x.cuda(), requires_grad=True)
        output = Variable(y.cuda())
        v = []
        for e in net.parameters():
            print(e.size())
            v.append(Variable(torch.FloatTensor(e.data.size()).normal_().cuda()))
        exact = Hv_exact2(net, loss_f, input, output, v)
        approx = Hv_approx(net, loss_f, input, output, v)
        norm_diff = 0.0
        norm_sum = 0.0
        for k1, k2 in zip(exact, approx):
            error = k1.data - k2.data
            norm_diff += torch.sum(error * error)
            norm_sum += torch.sum(k1.data * k1.data)
        print(norm_diff / norm_sum)
        exit(0)
