import torch
import torch.functional as F
import numpy as np
import time
import os
import shutil

def remake_dir(out_dirPath):
    if os.path.exists(out_dirPath):
        shutil.rmtree(out_dirPath)
        print(f"have remaken {out_dirPath}")
    os.makedirs(out_dirPath)

def train(
    model,
    criterion,
    optimizer,
    batch_size,
    max_epoch=100,
    print_per_iter=10,
    train_loader=None,
    valid_loader=None,
    test_loader=None,
):

    train_loss = []
    valid_loss = []
    valid_acc = []

    train_number = 5
    ans_epoch = 0
    ans_loss = 0
    ans_acc = 0

    remake_dir(f"./checkpoints{train_number}")

    for epoch in range(max_epoch):
        epoch_st = time.time()
        tmp_train_loss = []
        tmp_valid_loss = []
        valid_logits = []
        valid_labels = []
        for iter, batch in enumerate(train_loader):
            iter_st = time.time()
            loss, logits = train_step(
                model,
                criterion,
                optimizer,
                batch[0].cuda(),
                torch.from_numpy(np.array(batch[1])).cuda(),
            )
            tmp_train_loss.append(loss)
            if iter % print_per_iter == 0:
                print("train iter[{:3d}] | epoch[{:3d}] loss:{:.6f} time:{:.6f}".format(iter,epoch,loss,time.time()-iter_st))
        train_loss.append(np.mean(tmp_train_loss))
        print("train epoch[{:3d}] avg.loss:{:.6f} time:{:.6f}".format(epoch, train_loss[-1], time.time()-epoch_st))

        if valid_loader is not None:
            for iter, batch in enumerate(valid_loader):
                loss, logits = valid_step(
                    model,
                    criterion,
                    batch[0].cuda(),
                    torch.from_numpy(np.array(batch[1])).cuda(),
                )
                valid_logits.extend(logits.cpu().numpy())
                valid_labels.extend(batch[1])
                tmp_valid_loss.append(loss)
            valid_loss.append(np.mean(tmp_valid_loss))
            valid_acc.append(acc(valid_logits, valid_labels))
            print("valid epoch[{:3d}] avg.loss:{:.6f}  acc:{:.2%}\n".format(epoch, loss, valid_acc[-1]))
            torch.save(model, f"./checkpoints{train_number}/"+str(epoch)+".pt")
            if valid_acc[-1] > ans_acc:
                ans_epoch = epoch
                ans_loss = loss
                ans_acc = valid_acc[-1]


    if test_loader is not None:
        tmp_test_loss = []
        test_logits = []
        test_labels = []
        for iter, batch in enumerate(test_loader):
            loss, logits = test_step(
                model,
                criterion,
                batch[0].cuda(),
                torch.from_numpy(np.array(batch[1])).cuda(),
            )
            test_logits.extend(logits.cpu().numpy())
            test_labels.extend(batch[1])
            tmp_test_loss.append(loss)
        test_loss = np.mean(tmp_test_loss)
        print("test loss:", test_loss, "acc:", acc(test_logits, test_labels))

    print(f"the best model is in epoch{ans_epoch}, loss is {ans_loss}, acc is {ans_acc}")
    return train_loss, valid_loss, test_loss


def train_step(model, criterion, optimizer, data, label):
    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits, label)
    loss.backward()
    optimizer.step()
    return loss.item(), logits


def valid_step(model, criterion, data, label):
    with torch.no_grad():
        logits = model(data)
        loss = criterion(logits, label)
    return loss.item(), logits


def test_step(model, criterion, data, label):
    with torch.no_grad():
        logits = model(data)
        loss = criterion(logits, label)
    return loss.item(), logits

def acc(logits, label):
    # print(logits)
    pred = np.argmax(logits, axis=-1)
    right = 0
    for i in range(len(pred)):
        if label[i] == pred[i]:
            right = right + 1
    return right * 1.0 / len(pred)
