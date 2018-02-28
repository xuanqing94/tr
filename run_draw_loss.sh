#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 ./draw_loss.py --ngpu 4 --modelIn1 ./vgg16/b1024/sgd100.pth --modelIn2 ./vgg16/b1024/tr100_none.pth > ./vgg16/b1024/loss_cut_SGD_TR.txt
