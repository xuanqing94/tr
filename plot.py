#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def parse_file(fn):
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
    time_sgd, tr_loss_sgd, tr_accu_sgd, te_loss_sgd, te_accu_sgd = \
            parse_file('./vgg16/log_sgd_liu.txt')
    time_tr, tr_loss_tr, tr_accu_tr, te_loss_tr, te_accu_tr = \
            parse_file('./vgg16/log_tr_liu.txt')
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.semilogy(range(len(time_sgd)), tr_loss_sgd / 50000, 'r-', label='SGD test loss')
    ax1.semilogy(range(len(time_tr)), tr_loss_tr / 50000, 'b-', label='TR test loss')
    ax1_ = ax1.twinx()
    ax1_.semilogy(range(len(time_sgd)), 1.0 - tr_accu_sgd, 'r--', label='SGD test accuracy')
    ax1_.semilogy(range(len(time_tr)), 1.0 - tr_accu_tr, 'b--', label='TR test accuracy')


    ax2.plot(range(len(time_sgd)), te_accu_sgd, 'r-', label='SGD test accuracy')
    ax2.plot(range(len(time_sgd)), tr_accu_sgd, 'r--', label='SGD train accuracy')
    ax2.plot(range(len(time_tr)), te_accu_tr, 'b-', label='TR test accuracy')
    ax2.plot(range(len(time_tr)), tr_accu_tr, 'b--', label='TR train accuracy')
    ax2.legend(loc=0)

    plt.show()
