#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def fn(x):
    if x > -1.0:
        return 0.5 * x * x
    else:
        return 50.0 / 9 * (x + 1.3) ** 2

def plot_SGD_TR_SGD():
    x = np.arange(-1.6, 1.0, 0.01)
    y = np.array([fn(x_) for x_ in x])
    fd = { 'weight': 'bold', 'size': 12 }
    plt.axis('off')

    plt.plot(-1.6, fn(-1.6), 'ko')
    plt.plot(-1.3, fn(-1.3), 'ko')
    plt.plot(-1.0, fn(-1.0), 'ko')
    plt.plot(-0.3, fn(-0.3), 'ko')
    plt.plot(0.0, fn(0.0), 'ko')

    style="Simple,tail_width=0.5,head_width=4,head_length=8"
    kw1 = dict(arrowstyle=style, color="k", linestyle='--', linewidth=.5)
    kw2 = dict(arrowstyle=style, color="k", linestyle='--', linewidth=.5)

    a0 = patches.FancyArrowPatch((-1.6, fn(-1.6)), (-1.3, fn(-1.3)), connectionstyle="arc3", **kw1)
    a1 = patches.FancyArrowPatch((-1.3, fn(-1.3)), (-1.0, fn(-1.0)), connectionstyle="arc3", **kw1)
    a2 = patches.FancyArrowPatch((-1.0, fn(-1.0)), (-0.3, fn(-0.3)), connectionstyle="arc3", **kw1)
    a3 = patches.FancyArrowPatch((-0.3, fn(-0.3)), (0.0, fn(0.0)), connectionstyle="arc3", **kw1)

    for p in [a0, a1, a2, a3]:
        plt.gca().add_patch(p)
    plt.text(-1.5, fn(-1.5), "SGD", fontdict=fd)
    plt.text(-1.2, fn(-1.1), "TR", fontdict=fd)
    plt.text(-0.8, fn(-0.8), "TR", fontdict=fd)
    plt.text(-0.2, fn(-0.2), "SGD", fontdict=fd)
    plt.plot(x, y, color='b', linewidth=2)
    plt.show()


def plot_TR_SGD_TR():
    x = np.arange(-1.6, 1.0, 0.01)
    y = np.array([fn(x_) for x_ in x])
    fd = { 'weight': 'bold', 'size': 12 }
    plt.axis('off')

    plt.plot(1.0, fn(1.0), 'ko')
    plt.plot(0.3, fn(0.3), 'ko')
    plt.plot(0.0, fn(0.0), 'ko')
    plt.plot(-0.3, fn(-0.3), 'ko')

    style="Simple,tail_width=0.5,head_width=4,head_length=8"
    kw1 = dict(arrowstyle=style, color="k", linestyle='--', linewidth=.5)
    kw2 = dict(arrowstyle=style, color="k", linestyle='--', linewidth=.5)

    a0 = patches.FancyArrowPatch((1.0, fn(1.0)), (0.3, fn(0.3)), connectionstyle="arc3", **kw1)
    a1 = patches.FancyArrowPatch((0.3, fn(0.3)), (0.0, fn(0.0)), connectionstyle="arc3", **kw1)
    a2 = patches.FancyArrowPatch((0.0, fn(0.0)), (-0.3, fn(-0.3)), connectionstyle="arc3", **kw1)

    for p in [a0, a1, a2]:
        plt.gca().add_patch(p)
    plt.text(0.5, fn(0.5)+0.1, "TR", fontdict=fd)
    plt.text(0.07, fn(0.15), "SGD", fontdict=fd)
    plt.text(-0.15, fn(0.15), "TR", fontdict=fd)
    plt.plot(x, y, color='b', linewidth=2)
    plt.show()



if __name__ == "__main__":
    plot_SGD_TR_SGD()
