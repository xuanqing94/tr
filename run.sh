#!/usr/bin/env bash


net=vgg16
b=1024
dir=./${net}/b${b}
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --method tr --batchSize ${b} --epoch 100 --lr 0.5 --max_inner 0 --radius 2 --ngpu 4 --modelOut sgd_0.5_100.pth > log_sgd_0.5_100.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --method tr --batchSize ${b} --epoch 30 --lr 0.1 --max_inner 0 --radius 2 --ngpu 4 --modelIn sgd_0.5_100.pth --modelOut sgd__0.5_100_0.1_30.pth > log_sgd_0.5_100_0.1_30.txt
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --method tr_full_grad --batchSize ${b} --epoch 100 --lr 0.1 --max_inner 1 --radius 2 --ngpu 4 --modelOut tr_full_grad.pth > log_tr_full_grad.txt
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --method adam --batchSize ${b} --epoch 20 --lr 1 --max_inner 0 --radius 2 --ngpu 4 --modelOut ${dir}/adam100_lr1.pth > ${dir}/log_adam100_lr1.txt
