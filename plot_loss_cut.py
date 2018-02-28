#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def read_f(f):
    alphas = []
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for l in open(f):
        if l[0] == '#':
            continue
        items = l.split(',')
        alphas.append(float(items[0]))
        train_loss.append(float(items[1]))
        train_acc.append(float(items[2]))
        test_loss.append(float(items[3]))
        test_acc.append(float(items[4]))
    return alphas, train_loss, np.array(train_acc) * 100, test_loss, np.array(test_acc) * 100


if __name__ == "__main__":
    lbfont = 20
    titfont = 20
    txtfont = 16
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    alphas, train_loss, train_acc, test_loss, test_acc = read_f("./vgg16/b1024/loss_cut_Hybrid_TR.txt")
    ax1.plot(alphas, train_loss, color='b', linestyle='-', label='Train loss')
    ax1.plot(alphas, test_loss, color='b', linestyle=':', label='Test loss')
    ax1.plot(np.nan, color='r', linestyle='-', label='Train accuracy')
    ax1.plot(np.nan, color='r', linestyle=':', label='Test accuracy')
    ax1.axvline(0.0, color='k')
    ax1.axvline(1.0, color='k')
    ax1.set_xlabel('alpha', fontsize=lbfont, fontweight='bold')
    ax1.set_ylabel('Loss', color='b', fontsize=lbfont, fontweight='bold')
    ax1.text(0.0, 0.004, 'TR', fontsize=txtfont)
    ax1.text(1.0, 0.004, 'Hybrid', fontsize=txtfont)
    ax1.legend(loc=0, prop={'size': 16})
    ax1_ = ax1.twinx()
    ax1_.plot(alphas, train_acc, color='r', linestyle='-')
    ax1_.plot(alphas, test_acc, color='r', linestyle=':')
    #ax1_.set_ylabel('Accuracy(%)', color='r', fontsize=14)
    ax1_.set_title("Dist(Hybrid, TR)=41.02", fontsize=titfont, fontweight='bold')
    #------------------------------------------------------
    alphas, train_loss, train_acc, test_loss, test_acc = read_f("./vgg16/b1024/loss_cut_SGD_TR.txt")
    ax2.plot(alphas, train_loss, color='b', linestyle='-', label='Train loss')
    ax2.plot(alphas, test_loss, color='b', linestyle=':', label='Test loss')
    ax2.plot(np.nan, color='r', linestyle='-', label='Train accuracy')
    ax2.plot(np.nan, color='r', linestyle=':', label='Test accuracy')
    ax2.axvline(0.0, color='k')
    ax2.axvline(1.0, color='k')
    ax2.set_xlabel('alpha', fontsize=lbfont, fontweight='bold')
    #ax2.set_ylabel('Loss', color='b', fontsize=14)
    ax2.text(0.0, 0.026, 'TR', fontsize=txtfont)
    ax2.text(1.0, 0.026, 'SGD', fontsize=txtfont)
    ax2.legend(loc=2, prop={'size': 16})
    ax2_ = ax2.twinx()
    ax2_.plot(alphas, train_acc, color='r', linestyle='-')
    ax2_.plot(alphas, test_acc, color='r', linestyle=':')
    #ax2_.set_ylabel('Accuracy(%)', color='r', fontsize=14)
    ax2_.set_title("Dist(TR, SGD)=3341.19", fontsize=titfont, fontweight='bold')
    #-----------------------------------------------------
    alphas, train_loss, train_acc, test_loss, test_acc = read_f("./vgg16/b1024/loss_cut_SGD_Hybrid.txt")
    ax3.plot(alphas, train_loss, color='b', linestyle='-', label='Train loss')
    ax3.plot(alphas, test_loss, color='b', linestyle=':', label='Test loss')
    ax3.plot(np.nan, color='r', linestyle='-', label='Train accuracy')
    ax3.plot(np.nan, color='r', linestyle=':', label='Test accuracy')
    ax3.axvline(0.0, color='k')
    ax3.axvline(1.0, color='k')
    ax3.set_xlabel('alpha', fontsize=lbfont, fontweight='bold')
    #ax3.set_ylabel('Loss', color='b', fontsize=14)
    ax3.text(0.0, 0.026, 'Hybrid', fontsize=txtfont)
    ax3.text(1.0, 0.026, 'SGD', fontsize=txtfont)
    ax3.legend(loc=2, prop={'size': 16})
    ax3_ = ax3.twinx()
    ax3_.plot(alphas, train_acc, color='r', linestyle='-')
    ax3_.plot(alphas, test_acc, color='r', linestyle=':')
    ax3_.set_ylabel('Accuracy(%)', color='r', fontsize=lbfont, fontweight='bold')
    ax3_.set_title("Dist(Hybrid, SGD)=3351.84", fontsize=titfont, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1_.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2_.tick_params(axis='both', which='major', labelsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3_.tick_params(axis='both', which='major', labelsize=16)
    plt.show()
