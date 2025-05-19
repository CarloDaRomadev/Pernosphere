'''

train_test.py

This file contains the function for train and test the model.

'''

import torch
from torch.utils.data import DataLoader
from torch import nn
import torchmetrics
from dataloader import MyDataloader
import random
import pandas as pd
import matplotlib.pyplot as plt
from model import OurCNN

if __name__ == "__main__":

    try:
        if torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch.version, "hip") and torch.version.hip:
            dev = "hip"
        else:
            dev = "cpu"
    except AttributeError:
        dev = "cpu"

    print(f"Best device for your config is: {dev}")
    

    seed = random.randint(0, 1000)
    train_ds = MyDataloader("AI-Lab_project/Dataset/dataset_leaf_labeled_train.csv", "AI-Lab_project/Archive/all_leaf_train", transform=None, seed=seed)
    test_ds = MyDataloader("AI-Lab_project/Dataset/dataset_leaf_labeled_test.csv", "AI-Lab_project/Archive/all_leaf_test", transform=None, seed=seed)

    '''
    
    This training function includes early stopping: if the test 
    accuracy doesn't improve for a "patience" epochs, the
    training process stops automatically.
    (usually 9/10 Epochs are sufficent)

    '''

    bs = 16
    lr = 0.0005
    max_epochs = 9999

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=bs)

    model = OurCNN().to(dev)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    acc_train = torchmetrics.Accuracy(task='multiclass', num_classes=4).to(dev)
    acc_test = torchmetrics.Accuracy(task='multiclass', num_classes=4).to(dev)

    best_acc = 0.0
    wait = 0
    patience = 3

    losses = []
    accs_train = []
    accs_test = []

    for ep in range(max_epochs):
        print(f"EPOCH: {ep+1}/{max_epochs}")

        model.train()
        total_loss = 0

        for b, (x, y) in enumerate(train_dl):
            x, y = x.to(dev), y.to(dev)
            optim.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            acc_train.update(out, y)

            if b % 50 == 0:
                print(f"B: {b}, L: {loss.item():.4f}")

        ep_acc = acc_train.compute()
        acc_train.reset()
        avg_loss = total_loss / len(train_dl)
        print(f"L: {avg_loss:.4f}, TRAIN ACCURACY: {ep_acc:.4f}")

        losses.append(avg_loss)
        accs_train.append(ep_acc.item())

        model.eval()
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(dev), y.to(dev)
                out = model(x)
                acc_test.update(out, y)

        val_acc = acc_test.compute()
        acc_test.reset()
        print(f"TEST ACCURACY: {val_acc:.4f}")
        accs_test.append(val_acc.item())

        if val_acc > best_acc:
            best_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), 'AI-Lab_project/Model/model_leaf.pth')
            print("Model correctly SAVED in AI-Lab_project/Model/model_leaf.pth'")
        else:
            wait += 1

        if wait >= patience:
            print("Stop")
            break

    print("Finished")

    # Plotting
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accs_train, label='Train Acc')
    plt.plot(epochs, accs_test, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
