#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def read_f(f):
    time = []
    acc = []
    loss = []
    test_acc = []
    for line in open(f):
        if line[0] != '[':
            continue
        items = line.split(',')
        time.append(float(items[0].split(':')[1]))
        loss.append(float(items[1].split(':')[1]))
        acc.append(float(items[2].split(':')[1]))
        test_acc.append(float(items[4].split(':')[1]))
    return np.array(time), np.array(loss), np.array(acc), np.array(test_acc)

if __name__ == "__main__":

    plt.subplot(131)
    t_1, _, _, ta_1 = read_f('log_sgd_1_40.txt')
    t_05, _, _, ta_05 = read_f('log_sgd_0.5_100.txt')
    t_01, _, _, ta_01 = read_f('./vgg16/b1024/log_sgd_liu_1.txt')
    t_005, _, _, ta_005 = read_f('log_sgd_0.05_40.txt')
    plt.plot(100 - 100 * ta_1[:30], label='$\eta=1$')
    plt.plot(100 - 100 * ta_05[:30], label='$\eta=0.5$')
    plt.plot(100 - 100 * ta_01[:30], label='$\eta=0.1$')
    plt.plot(100 - 100 * ta_005[:30], label='$\eta=0.05$')

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Test error (%)')



    plt.subplot(132)
    #t_1, _, _, ta_1 = read_f('log_sgd_1_40.txt')
    t_05, _, _, ta_05 = read_f('log_sgd_0.5_100.txt')
    t_01, _, _, ta_01 = read_f('./vgg16/b1024/log_sgd_liu_1.txt')
    #t_005, _, _, ta_005 = read_f('log_sgd_0.05_40.txt')
    tr, _, _, tr_acc = read_f('./vgg16/b1024/log_tr_liu_2.0_1.txt')
    #plt.plot(100 - 100 * ta_1, label='$\eta=1$')
    plt.plot(100 - 100 * ta_05, label='$\eta=0.5$')
    plt.plot(100 - 100 * ta_01, label='$\eta=0.1$')
    #plt.plot(100 - 100 * ta_005, label='$\eta=0.05$')
    plt.plot(100 - 100 * tr_acc, label='TR')

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Test error (%)')


    plt.subplot(133)
    _, _, _, ta_05 = read_f('log_sgd_0.5_100.txt')
    _, _, _, ta_01 = read_f('log_sgd_0.5_100_0.1_30.txt')
    plt.plot(range(len(ta_05)), 100 - 100 * ta_05, label='$\eta=0.5$')
    plt.plot(range(len(ta_05), len(ta_05) + len(ta_01)), 100 - 100 * ta_01, label='$\eta=0.1$')
    plt.xlabel('Epoch')
    plt.ylabel('Test error (%)')
    plt.gca().axvline(100, linewidth=2, linestyle='--')
    plt.legend()

    plt.show()
