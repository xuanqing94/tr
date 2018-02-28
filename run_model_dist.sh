#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 ./model_dist.py --ngpu 4 --modelIn1 ./vgg16/b1024/tr100_none.pth --modelIn2 ./vgg16/b1024/tr100_sgd50.pth
