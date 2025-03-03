
import torch
import numpy as np
from torch.nn import functional as F


def classification_net_accuracy(device, data_loader, model):

    # set model to eval mode
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for data, label in data_loader:

            # move data and label to device
            data, label = data.to(device), label.to(device)

            # forward pass
            logits = model(data)

            # compute softmax
            softmax = F.softmax(logits, dim=1)

            # get predictions and confidence values
            confidences, predictions = torch.max(softmax, dim=1)

            # update correct and total
            correct += (predictions == label).sum().item()
            total += label.size(0)

    # compute accuracy
    accuracy = (correct / total) * 100
    print(f'Test set accuracy: {accuracy:.2f}%')

    return accuracy