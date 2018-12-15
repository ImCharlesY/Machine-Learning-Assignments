#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : main
Author          : Charles Young
Python Version  : Python 3.6.2
Date            : 2018-12-01
'''

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
from __future__ import print_function

from io import StringIO
import os
import sys
import logging
import argparse
from datetime import datetime

import torch
import torch.optim as optim
from torchsummary import summary

from nets import *
from cnn_trainer import Trainer
from utils import transforms
from utils.load_dataset import load_dataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-d', default='./data', nargs='?', 
                    help='directory to store dataset. Default \'./data\'')
parser.add_argument('-l', default=None, nargs='?',
                    help='path to save log. Default \'log_<model_name>_<datetime>.txt\'')
# Checkpoints
parser.add_argument('-c', default=None, nargs='?', 
                    help='path to save checkpoint. Default \'ckpt_<model_name>_<datetime>.t7\'')
parser.add_argument('-r', default=None, nargs='?',
                    help='path to latest checkpoint. Default None')
# Architecture
parser.add_argument('-m', default='PreActResNet18', choices = ['VGG19', 'LeNet', 'PreActResNet18', 'MobileNetV2', 'WRN_28_10'], nargs='?', 
                    help='Model name to apply. Default WRN_28_10')
parser.add_argument('--depth', type=int, default=28, nargs='?', 
                    help='Model depth. Default 28')
parser.add_argument('--widen-factor', type=int, default=10, nargs='?',  
                    help='Widen factor. Default 10')
# Random Erasing
parser.add_argument('--rt', metavar='RANDOM_ERASING', action='store_const', const=True, 
                    help='Specify to apply Random Erasing')
parser.add_argument('--p', default=0.5, type=float, nargs='?', 
                    help='Random Erasing - probability. Default 0.5')
parser.add_argument('--sl', default=0.02, type=float, nargs='?', 
                    help='Random Erasing -min erasing area. Default 0.02')
parser.add_argument('--sh', default=0.4, type=float, nargs='?', 
                    help='Random Erasing - max erasing area. Default 0.4')
parser.add_argument('--r1', default=0.3, type=float, nargs='?', 
                    help='Random Erasing - aspect of erasing area. Default 0.3')
# Optimization options
parser.add_argument('--train-batch', default=128, type=int, nargs='?',  
                    help='train batchsize. Default 128')
parser.add_argument('--test-batch', default=100, type=int, nargs='?', 
                    help='test batchsize. Default 100')
parser.add_argument('--max-epoch', default=300, type=int, nargs='?', 
                    help='max epoch, default is 300')
parser.add_argument('--early-stopping', default=10, type=int, nargs='?', 
                    help='early stopping patience, default is 10')
parser.add_argument('--drop', default=0.0, type=float, nargs='?', 
                    help='dropout ratio. Default 0.0')
parser.add_argument('--lr', default=0.1, type=float, nargs='?', 
                    help='initial learning rate. Default 0.1')
parser.add_argument('--momentum', default=0.9, type=float, nargs='?', 
                    help='momentum. Default 0.9')
parser.add_argument('--weight-decay', default=1e-4, type=float, nargs='?', 
                    help='weight decay. Default 1e-4')
parser.add_argument('--schedule', type=int, default=[150, 225], nargs='*', metavar='epoch', 
                    help='decrease learning rate at these epochs. Default [150, 225]')
parser.add_argument('--gamma', type=float, default=0.1, nargs='?', 
                    help='LR is multiplied by gamma on schedule. Default 0.1')
args = parser.parse_args()

# Data
print('==> Preparing data..')
m_transforms = args.rt and [transforms.RandomErasing(p = args.p, sl = args.sl, sh = args.sh, r1 = args.r1)] or []
print(m_transforms)
trainloader, validloader, __, __ = load_dataset(directory=args.d, 
    train_batch_size=args.train_batch, 
    test_batch_size=args.test_batch, 
    extra_transforms=m_transforms)

# Model
print('==> Building model..')
netList = {
'LeNet':LeNet(),
'VGG19':VGG('VGG19'),
'PreActResNet18':PreActResNet18(),
'MobileNetV2':MobileNetV2(10),
'WRN_28_10':WideResNet(args.depth, 10, widen_factor=args.widen_factor, dropRate=0.0)
}

net = netList[args.m]
opt = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
sch = optim.lr_scheduler.MultiStepLR(opt, milestones=args.schedule, gamma=args.gamma)
model = Trainer(net, opt, sch=sch, max_epoch=args.max_epoch, early_stopping=args.early_stopping)

if args.c is None:
	args.c = 'ckpt_'+args.m+datetime.now().strftime("_%Y%m%d_%H%M%S")+'.t7'
if args.l is None:
    args.l = 'log_'+args.m+datetime.now().strftime("_%Y%m%d_%H%M%S")+'.txt'
if args.r is not None:
	print('==> Load pre-trained model..')
	model.resume(args.r)

# redirect stdout to string buffer
old_stdout = sys.stdout
sys.stdout = stringOut = StringIO()
summary(net.cuda(), (3,32,32))
# redirect back
sys.stdout = old_stdout
model_summary = stringOut.getvalue()
print(model_summary)

# Train
print('==> Training model..')
model.train(trainloader, validloader, checkpoint=args.c, log_file_name=args.l, model_summary = model_summary)

# rename log file
os.rename('./logs/'+args.l, './logs/'+args.l.replace('log', 'logDR' if args.rt else 'logD') )
