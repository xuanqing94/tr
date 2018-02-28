#!/usr/bin/env python3

import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt

def read_file(fn):
    time = []
    train_loss = []
    train_accu = []
    test_loss = []
    test_accu = []
    with open(fn, 'r') as fh:
        for line in fh:
            if line[0] != '[':
                continue
            items = line.split(',')
            time.append(float(items[0].split(':')[-1]))
            train_loss.append(float(items[1].split(':')[-1]))
            train_accu.append(float(items[2].split(':')[-1]))
            test_loss.append(float(items[3].split(':')[-1]))
            test_accu.append(float(items[4].split(':')[-1]))
    return np.array(time), np.array(train_loss), np.array(train_accu), \
            np.array(test_loss), np.array(test_accu)


if __name__ == "__main__":
    bsize = [2048, 1024, 512, 256, 128, 64]
    dname = ["./vgg16/b{}".format(b) for b in bsize]
    mean_test_acc_sgd = []
    mean_test_acc_tr = []
    mean_test_acc_hybrid = []

    std_test_acc_sgd = []
    std_test_acc_tr = []
    std_test_acc_hybrid = []

    mean_train_acc_sgd = []
    mean_train_acc_tr = []
    mean_train_acc_hybrid = []

    std_train_acc_sgd = []
    std_train_acc_tr = []
    std_train_acc_hybrid = []
    for d in dname:
        max_test_acc_sgd = []
        max_test_acc_tr = []
        max_test_acc_hybrid = []

        max_train_acc_sgd = []
        max_train_acc_tr = []
        max_train_acc_hybrid = []
        for f in os.listdir(d):
            fname = os.path.join(d, f)
            if f.startswith('log_sgd_liu_none'):
                _, _, train_acc, _, test_acc = read_file(fname)
                max_test_acc_hybrid.append(np.max(test_acc))
                max_train_acc_hybrid.append(np.max(train_acc))
            elif f.startswith('log_sgd_liu'):
                _, _, train_acc, _, test_acc = read_file(fname)
                max_test_acc_sgd.append(np.max(test_acc))
                max_train_acc_sgd.append(np.max(train_acc))
            elif f.startswith('log_tr_liu_2.0'):
                _, _, train_acc, _, test_acc = read_file(fname)
                max_test_acc_tr.append(np.max(test_acc))
                max_train_acc_tr.append(np.max(train_acc))

        mean_test_acc_sgd.append(np.mean(max_test_acc_sgd))
        std_test_acc_sgd.append(np.std(max_test_acc_sgd))
        mean_test_acc_tr.append(np.mean(max_test_acc_tr))
        std_test_acc_tr.append(np.std(max_test_acc_tr))
        mean_test_acc_hybrid.append(np.mean(max_test_acc_hybrid))
        std_test_acc_hybrid.append(np.std(max_test_acc_hybrid))

        mean_train_acc_sgd.append(np.mean(max_train_acc_sgd))
        std_train_acc_sgd.append(np.std(max_train_acc_sgd))
        mean_train_acc_tr.append(np.mean(max_train_acc_tr))
        std_train_acc_tr.append(np.std(max_train_acc_tr))
        mean_train_acc_hybrid.append(np.mean(max_train_acc_hybrid))
        std_train_acc_hybrid.append(np.std(max_train_acc_hybrid))

    #plt.xscale("log", nonposx="clip")
    plt.subplot(122)
    plt.errorbar(bsize, (1.0-np.array(mean_test_acc_sgd))*100, yerr=np.array(std_test_acc_sgd)*100, color='r', label='SGD')
    plt.errorbar(bsize, (1.0-np.array(mean_test_acc_tr))*100, yerr=np.array(std_test_acc_tr)*100, color='b', label='TR')
    plt.errorbar(bsize, (1.0-np.array(mean_test_acc_hybrid))*100, yerr=np.array(std_test_acc_hybrid)*100, color='c', label='Hybrid')

    print(mean_test_acc_sgd)
    print(mean_test_acc_tr)
    print(mean_test_acc_hybrid)
    plt.xlabel('Batch size')
    plt.ylabel('Test Error(%)')
    plt.legend()
    plt.subplot(121)
    plt.errorbar(bsize, (1.0-np.array(mean_train_acc_sgd))*100, yerr=np.array(std_train_acc_sgd)*100, color='r', label='SGD')
    plt.errorbar(bsize, (1.0-np.array(mean_train_acc_tr))*100, yerr=np.array(std_train_acc_tr)*100, color='b', label='TR')
    plt.errorbar(bsize, (1.0-np.array(mean_train_acc_hybrid))*100, yerr=np.array(std_train_acc_hybrid)*100, color='c', label='Hybrid')
    plt.xlabel('Batch size')
    plt.ylabel('Train Error(%)')
    plt.legend()
    plt.show()

