
import argparse
import os
import sys

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import tqdm

from config import cfg
from train import *

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running command:", str(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mode",
        help="Training mode or testing mode",
        type=str,
        choices=["train", "test"],
        default='train',
    )
    parser.add_argument(
        "-arch",
        help="Input MLP-Mixer model architecture.",
        type=str,
        choices=["B_16", "L_16", "Mixer-B_16_imagenet1k", "Mixer-L_16_imagenet1k"],
        default='B_16',
    )
    parser.add_argument(
        "-cp",
        help="Input checkpoints path.",
        default=None,
    )
    parser.add_argument(
        "-dataset",
        help="Specify dataset",
        choices=["cifar10", "cifar100"],
        default="cifar10",
    )
    parser.add_argument(
        "-E",
        "--epochs",
        type=int,
        help="Maxium number of epochs to train.",
        default=5,
    )
    parser.add_argument(
        "-LR",
        "--learning_rate",
        type=float,
        help="Learning rate for training input transformation of training clean model.",
        default=5,
    )
    parser.add_argument(
        "-BS",
        "--batch-size",
        type=int,
        help="Training batch size.",
        default=128,
    )
    parser.add_argument(
        "-TBS",
        "--test-batch-size",
        type=int,
        help="Test batch size.",
        default=100,
    )
    parser.add_argument(
        "-pretrained",
        type=bool,
        help="Loading pertrained model or not.",
        default=True,
    )
    parser.add_argument(
        "-finetune",
        type=bool,
        help="Finetune the model or not.",
        default=True,
    )

    args = parser.parse_args()
    cfg.epochs = args.epochs
    cfg.learning_rate = args.learning_rate
    cfg.batch_size = args.batch_size
    cfg.test_batch_size = args.test_batch_size

    print("Preparing data..", args.dataset)
    if args.dataset == "cifar10":
        dataset = "cifar10"
        classes = 10
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((224, 224)), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root=cfg.data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.CIFAR10(
            root=cfg.data_dir,
            train=False,
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )
    elif args.dataset == "cifar100":
        dataset = "cifar100"
        classes = 100
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.CIFAR100(
            root=cfg.data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.CIFAR100(
            root=cfg.data_dir,
            train=False,
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )
    
    if args.mode == 'train':
        # Start training the model with specific dataset.
        print('========== Start training the model with specific dataset. Pretrained: {}, Finetune: {} =========='.format(args.pretrained, args.finetune))
        training(trainloader, args.arch, args.cp, dataset, classes, args.pretrained, args.finetune, device)
    if args.mode == 'test':
        # Start testing the model with specific dataset.
        print('========== Start testing the model with specific dataset. ==========')
        training(trainloader, testloader, args.arch, args.cp, dataset, classes, device)

if __name__ == "__main__":
    main()