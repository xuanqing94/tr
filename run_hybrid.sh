#!/usr/bin/env bash

net=densenet-cifar10
bs=128

# Experiment A.
# Route 1. TR(epoch=100) --> SGD(epoch=50) --> TR(epoch=50)
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --batchSize ${bs} --epoch 200 --lr 1.0e-1 --max_inner 1 --radius 1 --ngpu 4 --modelOut ./${net}/b${bs}/tr100.pth > ./${net}/b${bs}/log_tr100.txt
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --batchSize ${bs} --epoch 50 --lr 1.0e-1 --max_inner 0 --radius 2 --ngpu 4 --modelIn ./${net}/b${bs}/tr300.pth --modelOut ./${net}/b${bs}/tr300_sgd50.pth > ./${net}/b${bs}/log_tr300_sgd50.txt
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --batchSize ${bs} --epoch 150 --lr 1.0e-1 --max_inner 1 --radius 2 --ngpu 4 --modelIn ./${net}/b${bs}/tr300_sgd50.pth --modelOut ./${net}/b${bs}/tr300_sgd50_tr150.pth > ./${net}/b${bs}/log_tr300_sgd50_tr150.txt

# Route 2. SGD(epoch=50) --> TR(epoch=100) --> SGD(epoch=50)
CUDA_VISIBLE_DEVICES=3 ./main.py --batchSize ${bs} --epoch 150 --lr 1.0e-1 --max_inner 0 --radius 2 --ngpu 1 --method momsgd --modelOut ./${net}/b${bs}/sgd150.pth > ./${net}/b${bs}/log_sgd150.txt
CUDA_VISIBLE_DEVICES=3 ./main.py --batchSize ${bs} --epoch 100 --lr 1.0e-2 --max_inner 0 --radius 2 --ngpu 1 --method momsgd --modelIn ./${net}/b${bs}/sgd150.pth --modelOut ./${net}/b${bs}/sgd250.pth > ./${net}/b${bs}/log_sgd250.txt
CUDA_VISIBLE_DEVICES=3 ./main.py --batchSize ${bs} --epoch 100 --lr 1.0e-3 --max_inner 0 --radius 2 --ngpu 1 --method momsgd --modelIn ./${net}/b${bs}/sgd250.pth --modelOut ./${net}/b${bs}/sgd350.pth > ./${net}/b${bs}/log_sgd350.txt
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --batchSize ${bs} --epoch 300 --lr 1.0e-1 --max_inner 1 --radius 2 --ngpu 4 --modelIn ./${net}/b${bs}/sgd150.pth --modelOut ./${net}/b${bs}/sgd150_tr300.pth > ./${net}/b${bs}/log_sgd150_tr300.txt
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --batchSize ${bs} --epoch 50 --lr 1.0e-1 --max_inner 0 --radius 2 --ngpu 4 --modelIn ./${net}/b${bs}/sgd150_tr300.pth --modelOut ./${net}/b${bs}/sgd150_tr300_sgd50.pth > ./${net}/b${bs}/log_sgd150_tr300_sgd50.txt


# Experiment B.
# Route 1. SGD(lr=0.1, epoch=50) -> SGD(lr=0.01, epoch=50)
#CUDA_VISIBLE_DEVICES=1,2,3 ./main.py --batchSize 2048 --epoch 50 --lr 1.0e-2 --max_inner 0 --radius 2 --ngpu 3 --modelIn ./${net}/b2048/sgd50.pth --modelOut ./${net}/b2048/sgd50_lr0.01.pth > ./${net}/b2048/log_sgd50_lr0.01.txt

# Route 2. TR(lr=0.1, epoch=100) -> SGD(lr=0.1, epoch=50) -> SGD(lr=0.01, epoch=50)
#CUDA_VISIBLE_DEVICES=1,2,3 ./main.py --batchSize 2048 --epoch 50 --lr 1.0e-2 --max_inner 0 --radius 2 --ngpu 3 --modelIn ./${net}/b2048/tr100_sgd50.pth --modelOut ./${net}/b2048/tr100_sgd50_lr0.01.pth > ./${net}/b2048/log_tr100_sgd50_lr0.01.txt


# Experiment C.
# Try different algorithms
#CUDA_VISIBLE_DEVICES=0,1 ./main.py --batchSize ${bs} --epoch 100 --lr 1.0e-2 --max_inner 1 --radius 2 --ngpu 2 --method adam > ./${net}/b${bs}/log_adam100.txt
#CUDA_VISIBLE_DEVICES=2,3 ./main.py --batchSize ${bs} --epoch 100 --lr 1.0e-3 --max_inner 1 --radius 2 --ngpu 1 --method adam --modelOut ./${net}/b${bs}/adam100.pth > ./${net}/b${bs}/log_adam100.txt

