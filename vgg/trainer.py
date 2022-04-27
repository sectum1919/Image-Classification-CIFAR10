import torch
import torch.functional as F
import numpy as np


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
    for epoch in range(max_epoch):
        tmp_train_loss = []
        tmp_valid_loss = []
        valid_logits = []
        valid_labels = []
        for iter, batch in enumerate(train_loader):
            loss, logits = train_step(
                model,
                criterion,
                optimizer,
                batch[0].cuda(),
                torch.from_numpy(np.array(batch[1])).cuda(),
            )
            tmp_train_loss.append(loss)
            if iter % print_per_iter == 0:
                print("train iter[", iter, "]|epoch[", epoch, "] loss:", loss)
        train_loss.append(np.mean(tmp_train_loss))
        print("train epoch[", epoch, "] avg.loss:", train_loss[-1])

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
            print("valid epoch[", epoch, "] avg.loss:", valid_loss[-1], "acc:", valid_acc[-1])
            torch.save(model, "./checkpoints/"+str(epoch)+".pt")

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
