
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

from model.MlpMixer import MLPMixer
from config import cfg

def training(trainloader, arch, cp, dataset, classes, pretrained, finetune, device):
    model = MLPMixer(arch)
    if pretrained:
        model.load_state_dict(torch.load(cp))
        if finetune:
            model.head = nn.Linear(model.head.in_features, classes)
            print('++++++++++ Start Finetune Mode ++++++++++')
   
    model = model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=cfg.decay)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

    for epoch in tqdm.tqdm(range(cfg.epochs)):

        running_loss = 0.0
        running_correct = 0
        for batch_id, (inputs, outputs) in enumerate(trainloader):

            inputs = inputs.to(device)
            outputs = outputs.to(device)

            optimizer.zero_grad()

            model_outputs = model(inputs)  

            _, preds = torch.max(model_outputs, 1)
            outputs = outputs.view(outputs.size(0))

            loss = nn.CrossEntropyLoss()(model_outputs, outputs)

            # Compute gradient of perturbed weights with perturbed loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            running_correct += torch.sum(preds == outputs.data)

        accuracy = running_correct.double() / (len(trainloader.dataset))
        print("For epoch: {}, loss: {:.6f}, accuracy: {:.5f}".format(epoch, running_loss / len(trainloader.dataset), accuracy))

        if (epoch+1)%20 == 0 or (epoch+1) == cfg.epochs:

            extra = ["MLPMixer", arch + 'Q', dataset, "p", str(cfg.precision), "model", "StepLR", str(epoch+1)]

            model_path = os.path.join(cfg.model_dir_mlpmixer, arch, dataset, "_".join(extra) + ".pth")

            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))

            if os.path.exists(model_path):
                print("Checkpoint already present ('%s')" % model_path)
                sys.exit(1)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": running_loss / batch_id,
                    "accuracy": accuracy,
                },
                model_path,
            )