#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def srelu(x, alpha=1.0):
    return (np.sqrt(x**2 + alpha) + x) / 2.0

def relu(x):
    return np.maximum(x, 0)

if __name__ == "__main__":
    x = np.arange(-3, 3, 0.01)
    alpha = [0.2, 0.6, 1.0]
    for a in alpha:
        plt.plot(x, srelu(x, a), label="alpha={}".format(a))
    plt.plot(x, relu(x), label="ReLU")
    plt.legend(loc=0)
    plt.show()
