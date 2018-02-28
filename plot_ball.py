#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_visible(False)
    ax.axis('off')

    theta = np.linspace(0, 2*np.pi, 100)
    r = 1
    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)
    ax.plot(x1, x2, linestyle='--')
    ax.set_aspect(1)
    plt.plot(0, 0, 'o', color='k', markersize=4)
    plt.plot(0, 0.5, 'o', color='k', markersize=4)
    ax.arrow(0, 0, 0, 0.93, head_width=0.03, head_length=0.07, fc='k', ec='k')
    ax.arrow(0, 0.5, 0.2, np.sqrt((r-0.07)**2 - 0.2**2) - 0.5, head_width=0.03, head_length=0.07, fc='k', ec='k')
    ax.text(0.04, 0.0, '$x_t$')
    ax.text(-0.05, 1.04, '$x_{t+1}^{GD}$')
    ax.text(0.2, np.sqrt((r+0.04)**2 - 0.2**2), '$x_{t+1}^{TR}$')
    ax.text(-0.15, 0.7, '$-g$')
    ax.text(0.12, 0.7, '$-(g+\lambda Hg)$')

    plt.show()
