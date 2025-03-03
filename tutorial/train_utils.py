'''
This module contains methods for training models with different loss functions.
'''

import torch
from torch.nn import functional as F
from torch import nn

def train_single_epoch(
        epoch,
        device,
        train_loader,
        model,
        optimizer,
        loss_function,
        args
    ):

    # set model to train mode
    model.train()
    
    log_interval = 10
    train_loss = 0
    num_samples = 0

    for batch_idx, (data, labels) in enumerate(train_loader):

        # move data and labels to device
        data = data.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        logits = model(data)
        
        # compute loss
        loss = loss_function(logits, labels)
        if args.loss_mean:
            loss = loss / len(data)

        # backward pass
        loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)

        # update parameters
        train_loss += loss.item()
        optimizer.step()

        # log training loss
        num_samples += len(data)
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = train_loss / num_samples
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

def test_single_epoch(
        epoch,
        device,
        test_val_loader,
        model,
        loss_function,
        args
    ):

    # set model to eval mode
    model.eval()

    loss = 0
    num_samples = 0
    
    with torch.no_grad():

        for batch_idx, (data, labels) in enumerate(test_val_loader):

            # move data and labels to device
            data = data.to(device)
            labels = labels.to(device)

            # forward pass
            logits = model(data)

            # compute loss
            loss += loss_function(logits, labels).item()
            num_samples += len(data)

    avg_loss = loss / num_samples
    print(f'======> Test set loss: {avg_loss:.4f}')
    return avg_loss