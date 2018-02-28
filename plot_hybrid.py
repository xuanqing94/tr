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
    model = "vgg16-stl10"
    batch = 1024
    sgd50_f = "./{}/b{}/log_sgd150.txt".format(model, batch)
    sgd50_tr100_f = "./{}/b{}/log_sgd150_tr300.txt".format(model, batch)
    sgd50_tr100_sgd50_f = "./{}/b{}/log_sgd150_tr300_sgd50.txt".format(model, batch)

    tr100_f = "./{}/b{}/log_tr300.txt".format(model, batch)
    tr100_sgd50_f = "./{}/b{}/log_tr300_sgd50.txt".format(model, batch)
    tr100_sgd50_tr50_f = "./{}/b{}/log_tr300_sgd50_tr150.txt".format(model, batch)

    _, loss_sgd50, _, _, acc_sgd50 = parse_file(sgd50_f)
    _, loss_sgd50_tr100, _, _, acc_sgd50_tr100 = parse_file(sgd50_tr100_f)
    _, loss_sgd50_tr100_sgd50, _, _, acc_sgd50_tr100_sgd50 = parse_file(sgd50_tr100_sgd50_f)

    _, loss_tr100, _, _, acc_tr100 = parse_file(tr100_f)
    _, loss_tr100_sgd50, _, _, acc_tr100_sgd50 = parse_file(tr100_sgd50_f)
    _, loss_tr100_sgd50_tr50, _, _, acc_tr100_sgd50_tr50 = parse_file(tr100_sgd50_tr50_f)

    acc_sgd_tr_sgd = np.concatenate((acc_sgd50, acc_sgd50_tr100, acc_sgd50_tr100_sgd50))
    acc_tr_sgd_tr = np.concatenate((acc_tr100, acc_tr100_sgd50, acc_tr100_sgd50_tr50))
    loss_sgd_tr_sgd = np.concatenate((loss_sgd50, loss_sgd50_tr100, loss_sgd50_tr100_sgd50))
    loss_tr_sgd_tr = np.concatenate((loss_tr100, loss_tr100_sgd50, loss_tr100_sgd50_tr50))

    lw = 2

    plt.subplot(211)
    plt.plot(range(len(acc_sgd_tr_sgd)), (1.0 - acc_sgd_tr_sgd) * 100, color='r', linewidth=lw, label='SGD-TR-SGD')
    plt.plot(range(len(acc_tr_sgd_tr)), (1.0 - acc_tr_sgd_tr) * 100, color='b', linewidth=lw, label='TR-SGD-TR')

    plt.vlines(x=149, ymin=0, ymax=100, color='k', linestyle='--')
    plt.vlines(x=299, ymin=0, ymax=100, color='k', linestyle='--')
    plt.vlines(x=349, ymin=0, ymax=100, color='k', linestyle='--')
    plt.vlines(x=449, ymin=0, ymax=100, color='k', linestyle='--')
    plt.vlines(x=499, ymin=0, ymax=100, color='k', linestyle='--')


    plt.ylabel('Test error rate(%)')
    plt.ylim(17, 100)
    plt.legend(loc=0)

    plt.subplot(212)
    plt.semilogy(range(len(loss_sgd_tr_sgd)), loss_sgd_tr_sgd, color='r', linewidth=lw, label='SGD-TR-SGD')
    plt.semilogy(range(len(loss_tr_sgd_tr)), loss_tr_sgd_tr, color='b', linewidth=lw, label='TR-SGD-TR')

    plt.axvline(x=149, color='k', linestyle='--')
    plt.axvline(x=299, color='k', linestyle='--')
    plt.axvline(x=349, color='k', linestyle='--')
    plt.axvline(x=449, color='k', linestyle='--')
    plt.axvline(x=499, color='k', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.legend(loc=0)

    plt.show()

