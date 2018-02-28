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

def Hv_exact(net, loss_f, input, output, v):
    loss  = loss_f(net(input), output)
    grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
    Hv = torch.autograd.grad(grad_params, net.parameters(), grad_outputs=v, only_inputs=True)
    return Hv

def Hv_approx(net, loss_f, input, output, v, scale=10000.0):
    for p, vi in zip(net.parameters(), v):
        p.data -= vi / scale
    loss  = loss_f(net(input), output)
    grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=False)
    for p, vi in zip(net.parameters(), v):
        p.data += 2*vi / scale
    #net.zero_grad()

    loss  = loss_f(net(input), output)
    grad_params_ = torch.autograd.grad(loss, net.parameters(), create_graph=False)
    Hv = []
    for g1, g2 in zip(grad_params, grad_params_):
        Hv.append((g2 - g1) / 10.0 * scale)
    for p, vi in zip(net.parameters(), v):
        p.data -= vi / scale
    return Hv

def Hv_approx2(grad_params, net, loss_f, input, output, v, scale=100000.0):
    #loss  = loss_f(net(input), output)
    #grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=False)
    for p, vi in zip(net.parameters(), v):
        p.data += vi / scale
    loss  = loss_f(net(input), output)
    grad_params_ = torch.autograd.grad(loss, net.parameters(), create_graph=False)
    Hv = []
    for g1, g2 in zip(grad_params, grad_params_):
        Hv.append((g2 - g1) * scale)
    for p, vi in zip(net.parameters(), v):
        p.data -= vi / scale
    return Hv

def iter_TR(vx, vy, net, loss_f, lr, radius, max_iter):
    loss = loss_f(net(vx), vy)
    #grad = torch.autograd.grad(loss, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
    grad = torch.autograd.grad(loss, net.parameters(), create_graph=False)
    solution = []
    for g in grad:
        s = g.data
        solution.append(-lr * s)
    for i in range(max_iter):
        # calculate Hv+g
        Hv_g = Hv_approx2(grad, net, loss_f, vx, vy, solution)
        for idx, val in enumerate(Hv_g):
            val.data += grad[idx].data
        # do one gradient descent
        for idx, val in enumerate(solution):
            val -= lr / 2 * Hv_g[idx].data
        # project onto sphere
        norm = 0.0
        for val in solution:
            tmp = val.cpu().numpy()
            x_ = np.linalg.norm(tmp)
            norm += x_ * x_
        norm = np.sqrt(norm)
        if True:
            for val in solution:
                val *= radius / (norm + 1.0e-9)
    # update variable
    for idx, w in enumerate(net.parameters()):
        w.data += solution[idx]

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

def train_TR(dataloader, dataloader_test, net, loss_f, lr, radius=1.0, max_iter=1, max_epoch=10):
    run_time = 0.0
    for epoch in range(max_epoch):
        beg = time.time()
        data_iter = iter(dataloader)
        for x, y in data_iter:
            vx = Variable(x).cuda()
            vy = Variable(y).cuda()
            iter_TR(vx, vy, net, loss_f, lr, radius, max_iter)
        run_time += time.time() - beg
        print("[Epoch {}] Time: {}, Train loss: {}, Train accuracy: {}, Test loss: {}, Test accuracy: {}".format(epoch, run_time, loss(dataloader, net, loss_f), accuracy(dataloader, net), loss(dataloader_test, net, loss_f), accuracy(dataloader_test, net)))


def train_other(dataloader, dataloader_test, net, loss_f, lr, name='adam', max_epoch=10):
    run_time = 0.0
    if name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0, 0))
    elif name == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=lr)
    elif name == 'momsgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5.0e-4)
    else:
        print('Not implemented')
        exit(-1)
    for epoch in range(max_epoch):
        beg = time.time()
        data_iter = iter(dataloader)
        for x, y in data_iter:
            vx, vy = Variable(x).cuda(), Variable(y).cuda()
            optimizer.zero_grad()
            lossv = loss_f(net(vx), vy)
            lossv.backward()
            optimizer.step()
        run_time += time.time() - beg
        print("[Epoch {}] Time: {}, Train loss: {}, Train accuracy: {}, Test loss: {}, Test accuracy: {}".format(epoch, run_time, loss(dataloader, net, loss_f), accuracy(dataloader, net), loss(dataloader_test, net, loss_f), accuracy(dataloader_test, net)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1.0e-3)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--radius', type=float, default=1.0)
    parser.add_argument('--max_inner', type=int, default=1)
    parser.add_argument('--modelIn', type=str, default=None)
    parser.add_argument('--modelOut', type=str, default=None)
    parser.add_argument('--method', type=str, default="tr")
    opt = parser.parse_args()
    print(opt)
    net = VGG("VGG16")
    #net = densenet_cifar()
    #net = GoogLeNet()
    #net = MobileNet(num_classes=100)
    #net = stl10(32)
    net = nn.DataParallel(net, device_ids=range(opt.ngpu))
    #net = Test()
    net.apply(weights_init)
    if opt.modelIn is not None:
        net.load_state_dict(torch.load(opt.modelIn))
    loss_f = nn.CrossEntropyLoss()
    net.cuda()
    loss_f.cuda()
    #data = dst.STL10(root="/home/luinx/data/stl10", split='train', download=True, transform=tfs.Compose(
    #    [ tfs.Pad(4), tfs.RandomCrop(96), tfs.Scale(32), tfs.RandomHorizontalFlip(), tfs.ToTensor(), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]))
    #data_test = dst.STL10(root="/home/luinx/data/stl10", split='test', download=True, transform=tfs.Compose(
    #    [ tfs.Pad(4), tfs.RandomCrop(96), tfs.Scale(32), tfs.RandomHorizontalFlip(), tfs.ToTensor(), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]))
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
    dataloader = DataLoader(data, batch_size=opt.batchSize, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(data_test, batch_size=opt.batchSize, shuffle=False, num_workers=2)
    if opt.method == 'tr':
        train_TR(dataloader, dataloader_test, net, loss_f, opt.lr, opt.radius, opt.max_inner, opt.epoch)
    else:
        train_other(dataloader, dataloader_test, net, loss_f, opt.lr, opt.method, opt.epoch)

    # save model
    if opt.modelOut is not None:
        torch.save(net.state_dict(), opt.modelOut)

if __name__ == "__main__":
   main()
