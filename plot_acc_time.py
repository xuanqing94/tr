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
    f_sgd = "./vgg16/b1024/log_sgd_lr1.0.txt"
    f_tr = "./vgg16/b1024/log_sgd.txt"

    time_sgd, train_loss_sgd, train_acc_sgd, test_loss_sgd, test_acc_sgd = parse_file(f_sgd)
    time_tr, train_loss_tr, train_acc_tr, test_loss_tr, test_acc_tr = parse_file(f_tr)

    ax1 = plt.subplot(241)

    ax1.plot(time_sgd, (1.0 - train_acc_sgd) * 100, color='r', label='SGD')
    ax1.plot(time_tr, (1.0 - train_acc_tr) * 100, color='b', label='TR')
    ax1.set_xlabel('Time(second)')
    ax1.set_ylabel('Train Error(%)')

    plt.ylim([0, 40])
    plt.legend()

    ax2 = plt.subplot(242)
    ax2.plot(time_sgd, (1.0 - test_acc_sgd) * 100, color='r', label='SGD')
    ax2.plot(time_tr, (1.0 - test_acc_tr) * 100, color='b', label='TR')
    ax2.set_xlabel('Time(second)')
    ax2.set_ylabel('Test Error(%)')
    plt.ylim([10, 40])
    plt.legend()

    ax3 = plt.subplot(243)
    ax3.plot(range(100), (1.0 - train_acc_sgd[:100]) * 100, color='r', label='SGD')
    ax3.plot(range(len(train_acc_tr)), (1.0 - train_acc_tr) * 100, color='b', label='TR')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Train Error(%)')
    plt.ylim([0, 40])
    plt.legend()

    ax4 = plt.subplot(244)
    ax4.plot(range(100), (1.0 - test_acc_sgd[:100]) * 100, color='r', label='SGD')
    ax4.plot(range(len(test_acc_tr)), (1.0 - test_acc_tr) * 100, color='b', label='TR')
    ax4.set_xlabel('Epoch')
    plt.ylim([10, 40])
    ax4.set_ylabel('Test Error(%)')
    plt.legend()
    '''
    f_sgd1 = "./vgg16-stl10/b1024/log_sgd150.txt"
    f_sgd2 = "./vgg16-stl10/b1024/log_sgd300.txt"
    f_tr = "./vgg16-stl10/b1024/log_tr100.txt"

    time_sgd, train_loss_sgd, train_acc_sgd, test_loss_sgd, test_acc_sgd = parse_file(f_sgd1)
    time_sgd2, train_loss_sgd2, train_acc_sgd2, test_loss_sgd2, test_acc_sgd2 = parse_file(f_sgd2)
    time_sgd2 += time_sgd[-1]
    time_sgd = np.concatenate((time_sgd, time_sgd2))
    train_acc_sgd = np.concatenate((train_acc_sgd, train_acc_sgd2))
    test_acc_sgd = np.concatenate((test_acc_sgd, test_acc_sgd2))
    time_tr, train_loss_tr, train_acc_tr, test_loss_tr, test_acc_tr = parse_file(f_tr)

    ax1 = plt.subplot(245)

    ax1.plot(time_sgd[:230], (1.0 - train_acc_sgd[:230]) * 100, color='r', label='SGD')
    ax1.plot(time_tr, (1.0 - train_acc_tr) * 100, color='b', label='TR')
    ax1.set_xlabel('Time(second)')
    ax1.set_ylabel('Train Error(%)')
    plt.ylim([0, 40])
    plt.legend()

    ax2 = plt.subplot(246)
    ax2.plot(time_sgd[:230], (1.0 - test_acc_sgd[:230]) * 100, color='r', label='SGD')
    ax2.plot(time_tr, (1.0 - test_acc_tr) * 100, color='b', label='TR')
    ax2.set_xlabel('Time(second)')
    ax2.set_ylabel('Test Error(%)')
    plt.ylim([35, 50])
    plt.legend()

    ax3 = plt.subplot(247)
    ax3.plot(range(200), (1.0 - train_acc_sgd[:200]) * 100, color='r', label='SGD')
    ax3.plot(range(len(train_acc_tr)), (1.0 - train_acc_tr) * 100, color='b', label='TR')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Train Error(%)')
    plt.ylim([0, 40])
    plt.legend()

    ax4 = plt.subplot(248)
    ax4.plot(range(200), (1.0 - test_acc_sgd[:200]) * 100, color='r', label='SGD')
    ax4.plot(range(len(test_acc_tr)), (1.0 - test_acc_tr) * 100, color='b', label='TR')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Test Error(%)')
    plt.ylim([35, 50])
    plt.legend()
    '''

    plt.show()
