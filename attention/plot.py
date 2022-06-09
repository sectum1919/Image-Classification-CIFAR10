from importlib.resources import contents
from re import A
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

def read_log(logfile):
    train_loss = []
    test_acc = []
    with open(logfile) as fp:
        for line in fp.readlines():
            line = line.strip()
            if "train epoch" in line:
                train_loss.append(float(line.split('avg.loss:')[1][:8]))
            if "valid epoch" in line:
                test_acc.append(float(line.split('acc:')[1][:5])/100)
    return train_loss, test_acc

def plot_loss_acc_in_one(logfile, savefile):
    train_loss, test_acc = read_log(logfile)
    loss = np.array(train_loss)
    acc = np.array(test_acc)
    x = [i for i in range(len(train_loss))]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    p1 = ax1.plot(x, loss, label='loss')
    ax1.legend()
    p2 = ax2.plot(x, acc, label='acc')
    ax2.legend()
    pl.xlabel('epoch')
    ax1.set_ylabel('loss', color='r')
    ax2.set_ylabel('acc', color='g')
    plt.savefig(savefile, bbox_inches='tight')


def plot_many_loss_in_one(logfile_list, colors, labels, savefile):
    loss = []
    for log in logfile_list:
        train_loss, test_acc = read_log(log)
        loss.append(np.array(train_loss))
    x = [i for i in range(len(train_loss))]
    fig, ax1 = plt.subplots()
    for i in range(len(loss)):
        p = ax1.plot(x, loss[i], colors[i], label=labels[i], linewidth=0.8)
    ax1.legend()
    # pl.xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    plt.savefig(savefile, bbox_inches='tight')


def plot_many_loss_acc_in_one(logfile_list, colors, labels, savefile, max_epoch=100):
    loss = []
    acc = []
    for log in logfile_list:
        train_loss, test_acc = read_log(log)
        loss.append(np.array(train_loss[:max_epoch]))
        acc.append(np.array(test_acc[:max_epoch]))
    x = [i for i in range(len(loss[0]))]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i in range(len(loss)):
        p = ax1.plot(x, loss[i], colors[i], label=labels[i], linewidth=0.8, linestyle='-.')
        p = ax2.plot(x,  acc[i], colors[i], label=labels[i], linewidth=0.8, linestyle='-')
    ax2.legend(loc='right')
    # pl.xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax2.set_ylabel('acc')
    plt.savefig(savefile, bbox_inches='tight', dpi=600)

logfile_list = [
    '/nfs/user/chenrenmiao/project/Image-Classification-CIFAR10/attention/vit0.log',
    '/nfs/user/chenrenmiao/project/Image-Classification-CIFAR10/attention/vit1.log',
    '/nfs/user/chenrenmiao/project/Image-Classification-CIFAR10/attention/vit2.log',
    '/nfs/user/chenrenmiao/project/Image-Classification-CIFAR10/attention/vit3.log',
    '/nfs/user/chenrenmiao/project/Image-Classification-CIFAR10/attention/vit4.log',
    '/nfs/user/chenrenmiao/project/Image-Classification-CIFAR10/attention/vit5.log',
]
colors=['green','blue','red','yellow','black','purple']
labels=['ViT0','ViT1','ViT2','ViT3','ViT4','ViT5']
savefile='./vits_loss_acc.png'
# plot_many_loss_in_one(logfile_list, colors, labels, savefile)
plot_many_loss_acc_in_one(logfile_list, colors, labels, savefile)
