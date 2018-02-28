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
    _, _, _, _, sharp_acc_en1 = parse_file("./vgg16/b2048/log_sgd50.txt")
    _, _, _, _, sharp_acc_en2 = parse_file("./vgg16/b2048/log_sgd50_lr0.01.txt")
    _, _, _, _, wide_acc_en1 = parse_file("./vgg16/b2048/log_tr100_sgd50.txt")
    _, _, _, _, wide_acc_en2 = parse_file("./vgg16/b2048/log_tr100_sgd50_lr0.01.txt")
    sharp_acc = np.concatenate((sharp_acc_en1, sharp_acc_en2))
    wide_acc = np.concatenate((wide_acc_en1, wide_acc_en2))
    plt.plot(range(len(sharp_acc)), (1.0 - sharp_acc) * 100)
    plt.plot(range(len(wide_acc)), (1.0 - wide_acc) * 100)
    plt.show()
