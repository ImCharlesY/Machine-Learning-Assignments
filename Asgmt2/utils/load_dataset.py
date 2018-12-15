#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : load_dataset
Author          : Charles Young
Python Version  : Python 3.6.2
Date            : 2018-12-01
'''

from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms

import os

class torsampleCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_inputs = 1
        self.num_targets = 1
        
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return self.dataset.__len__()

def load_dataset(directory = './data', train_batch_size = 128, test_batch_size = 100, transform = True, extra_transforms = []):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        + extra_transforms
    ) if transform else transforms.Compose([])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) if transform else transforms.Compose([])

    trainset = torsampleCIFAR10(torchvision.datasets.CIFAR10(root=directory, train=True, download=True, transform=transform_train))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    testset = torsampleCIFAR10(torchvision.datasets.CIFAR10(root=directory, train=False, download=True, transform=transform_test))
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader, trainset.dataset, testset.dataset