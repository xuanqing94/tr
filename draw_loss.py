#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dst
import torchvision.transforms as tfs
from models.dog32 import Dog32
from models.net import VGG, Test
from models.resnet import ResNet18
from models.densenet import densenet_cifar
from models.googlenet import GoogLeNet
from models.mobilenet import MobileNet
from models.stl10_model import stl10
from torch.utils.data import DataLoader
import time

def loss(dataloader, net, loss_f):
    data_iter = iter(dataloader)
    total_loss = 0.0
    count = 0
    for x, y in data_iter:
        vx = Variable(x, volatile=True).cuda()
        vy = Variable(y).cuda()
        total_loss += torch.sum(loss_f(net(vx), vy).data)
        count += y.size()[0]
    return total_loss / count

def accuracy(dataloader, net):
    data_iter = iter(dataloader)
    count = 0
    total = 0
    for x, y in data_iter:
        vx = Variable(x, volatile=True).cuda()
        tmp = torch.sum(torch.eq(y.cuda(), torch.max(net(vx).data, dim=1)[1]))
        count += int(tmp)
        total += y.size()[0]
    return count / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--modelIn1', type=str, required=True)
    parser.add_argument('--modelIn2', type=str, required=True)
    parser.add_argument('--batchSize', type=int, default=128)
    opt = parser.parse_args()
    net1 = VGG("VGG16")
    net2 = VGG("VGG16")
    net = VGG("VGG16")
    net1 = nn.DataParallel(net1, device_ids=range(opt.ngpu))
    net2 = nn.DataParallel(net2, device_ids=range(opt.ngpu))
    net = nn.DataParallel(net, device_ids=range(opt.ngpu))
    net1.load_state_dict(torch.load(opt.modelIn1))
    net2.load_state_dict(torch.load(opt.modelIn2))
    loss_f = nn.CrossEntropyLoss()
    net1.cuda()
    net2.cuda()
    net.cuda()
    transform_train = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data = dst.CIFAR10("~/data/cifar10-py", download=False, train=True, transform=transform_train)
    data_test = dst.CIFAR10("~/data/cifar10-py", download=False, train=False, transform=transform_test)
    assert data, data_test
    dataloader = DataLoader(data, batch_size=opt.batchSize, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(data_test, batch_size=opt.batchSize, shuffle=False, num_workers=2)

    alphas = np.arange(-1.5, 1.3, 0.02)
    print("#Alpha, train loss, train acc, test loss, test acc")
    for alpha in alphas:
        state = {}
        net2_state = net2.state_dict()
        for k, v in net1.state_dict().items():
            state[k] = alpha * v + (1 - alpha) * net2_state[k]
        #for e1, e2, e in zip(net1.parameters(), net2.parameters(), net.parameters()):
        #    e.data = float(alpha) * e1.data + float(1 - alpha) * e2.data
        net.load_state_dict(state)
        net.train()
        train_loss = loss(dataloader, net, loss_f)
        train_acc = accuracy(dataloader, net)
        net.eval()
        test_loss = loss(dataloader_test, net, loss_f)
        test_acc = accuracy(dataloader_test, net)
        print("{}, {}, {}, {}, {}".format(alpha, train_loss, train_acc, test_loss, test_acc))

if __name__ == "__main__":
    main()
