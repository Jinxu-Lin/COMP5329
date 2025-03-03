import argparse
import random
import json
import os
import csv

import torch
from torch import optim
import torch.utils.data

# Import dataloaders
import Data.cifar10 as cifar10

# Import network models
from Models.resnet import resnet50

# Import optimizers
from Optimizers.optimizer import sgd, adam

# Import loss functions
from Losses.loss import cross_entropy

# Import train and validation utilities
from train_utils import train_single_epoch, test_single_epoch

# Import validation metrics
from Metrics.metrics import classification_net_accuracy


dataset_num_classes = {
    'cifar10': 10,
}

dataset_loader = {
    'cifar10': cifar10,
}

models = {
    'resnet50': resnet50,
}

optimizers = {
    'sgd': sgd,
    'adam': adam,
}

losses = {
    'cross_entropy': cross_entropy,
}

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def loss_function_save_name(loss_function):
    res_dict = {
        'cross_entropy': 'cross_entropy',
    }

    res_str = res_dict[loss_function]
    return res_str


def parseArgs():
    # default values
    seed = 42

    log_interval = 50
    
    dataset = 'cifar10'
    dataset_root = './'
    train_batch_size = 128
    test_batch_size = 128

    model = "resnet50"

    optimiser = "sgd"
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    first_milestone = 150 
    second_milestone = 250

    loss = "cross_entropy"

    epoch = 350

    save_interval = 50
    save_loc = './'
    saved_model_name = "resnet50_cross_entropy_350.model"

    parser = argparse.ArgumentParser(
        description="Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # seed
    parser.add_argument("--seed", type=int, default=seed,
                        dest="seed", help='seed')
    
    # device
    parser.add_argument("--gpu", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)

    # log
    parser.add_argument("--log-interval", type=int, default=log_interval,
                        dest="log_interval", help="Log Interval on Terminal")
    
    # dataset & dataloader
    parser.add_argument("--dataset", type=str, default=dataset,
                        dest="dataset", help='dataset to train on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=True)
    parser.add_argument("--train-batch-size", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("--test-batch-size", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")

    # model
    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to train')

    # optimizer
    parser.add_argument("--optimiser", type=str, default=optimiser,
                        dest="optimiser", help='Choice of optimisation algorithm')
    parser.add_argument("--learning-rate", type=float, default=learning_rate,
                        dest="learning_rate", help='Learning rate')
    parser.add_argument("--momentum", type=float, default=momentum,
                        dest="momentum", help='Momentum')
    parser.add_argument("--nesterov-momentum", action="store_true", dest="nesterov",
                        help="Whether to use nesterov momentum in SGD")
    parser.set_defaults(nesterov=False)
    parser.add_argument("--weight-decay", type=float, default=weight_decay,
                        dest="weight_decay", help="Weight Decay")
    
    # learning rate scheduler
    parser.add_argument("--first-milestone", type=int, default=first_milestone,
                        dest="first_milestone", help="First milestone to change lr")
    parser.add_argument("--second-milestone", type=int, default=second_milestone,
                        dest="second_milestone", help="Second milestone to change lr")
    
    # loss function
    parser.add_argument("--loss", type=str, default=loss, dest="loss_function",
                        help="Loss function to be used for training")
    parser.add_argument("--loss-mean", action="store_true", dest="loss_mean",
                        help="whether to take mean of loss instead of sum to train")
    parser.set_defaults(loss_mean=False)

    # training
    parser.add_argument("--epoch", type=int, default=epoch, dest="epoch",
                        help='Number of training epochs')
    
    # save
    parser.add_argument("--save-interval", type=int, default=save_interval,
                        dest="save_interval", help="Save Interval on Terminal")
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to export the model')
    
    return parser.parse_args()


def main(args):

    # seed
    set_seed(args.seed)

    # device
    cuda = False
    if (torch.cuda.is_available() and args.gpu):
        cuda = True
    device = torch.device("cuda" if cuda else "cpu")

    # log
    save_loc = args.save_loc
    os.makedirs(save_loc, exist_ok=True)
    log_loc = os.path.join(save_loc, 'logs')
    os.makedirs(log_loc, exist_ok=True)
    log_name = args.model + '_' + args.loss_function + '_' + str(args.epoch) + '.csv'
    log_file = open(os.path.join(log_loc, log_name), 'w')
    writer = csv.writer(log_file)
    writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Test Loss', 'Test Accuracy'])

    # dataset & dataloader
    num_classes = dataset_num_classes[args.dataset]
    train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu
        )
    test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu
        )

    # model
    net = models[args.model](num_classes=num_classes)

    # optimizer
    opt_params = net.parameters()
    optimizer = optimizers[args.optimiser](opt_params, args)

    # learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[args.first_milestone, args.second_milestone], 
        gamma=0.1
    )

    # loss function
    loss_function = losses[args.loss_function]()

    # Train!
    best_test_acc = 0

    for epoch in range(args.epoch):

        scheduler.step()

        train_loss = train_single_epoch(epoch, device, train_loader, net, optimizer, loss_function, args=args)
        val_loss = test_single_epoch(epoch, device, val_loader, net, loss_function, args=args)
        test_loss = test_single_epoch(epoch, device, test_loader, net, loss_function, args=args)
        test_acc = classification_net_accuracy(device, test_loader, net)

        # save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f'New best accuracy: {best_test_acc:.2f}%')
            save_dir = os.path.join(args.save_loc, 'best_models')
            os.makedirs(save_dir, exist_ok=True)
            save_name = args.model + '_' + args.loss_function + str(epoch + 1) + '.model'
            torch.save(net.state_dict(), os.path.join(save_dir, save_name))

        # save checkpoint every save_interval epochs
        if (epoch + 1) % args.save_interval == 0:
            save_dir = os.path.join(args.save_loc, 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            save_name = args.model + '_' + args.loss_function + str(epoch + 1) + '.model'
            torch.save(net.state_dict(), os.path.join(save_dir, save_name))

        # log
        writer.writerow([epoch + 1, train_loss, val_loss, test_loss, test_acc])
    
    # close log file
    log_file.close()


if __name__ == "__main__":
    args = parseArgs()
    main(args)