#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : main
Author          : Charles Young
Python Version  : Python 3.6.2
Date            : 2018-12-01
'''

from __future__ import print_function

import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from nets import *
from utils.utils import progress_bar

class Trainer():

    def __init__(self, net, opt, sch=None, max_epoch=300, early_stopping=10):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            torch.backends.cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = opt
        self.scheduler = sch

        self.max_epoch = max_epoch 
        self.early_stopping = early_stopping
        self.terminate = False
        self.start_epoch = 0
        self.best_acc = 0
        self.best_epoch = 0

    def train(self, train_loader, valid_loader, checkpoint='ckpt.t7', log_file_name='log.txt', model_summary = None):

        if not os.path.isdir('logs'):
            os.mkdir('logs')
        logging.basicConfig(format='%(asctime)s :%(message)s', filename = './logs/'+log_file_name, filemode = 'w', level = logging.INFO)

        if model_summary is not None:
            logging.info(model_summary)

        self.trainloader = train_loader
        self.validloader = valid_loader

        start_time = datetime.now()
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.scheduler is not None:
                self.scheduler.step()
            self.train_epoch(epoch)
            self.test_epoch(epoch, checkpoint)
            if self.terminate:
                break
        time_elapsed = datetime.now() - start_time
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        print('Best accuracy : {} ({}/{})'.format(self.best_acc, self.best_epoch, self.max_epoch))
        logging.info('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
        logging.info('Best accuracy : {} ({}/{})'.format(self.best_acc, self.best_epoch, self.max_epoch))

    def resume(self, checkpoint='ckpt.t7'):

        checkpoint = torch.load('./checkpoint/'+checkpoint)

        self.net.load_state_dict(checkpoint['net'])
        self.criterion.load_state_dict(checkpoint['cri'])
        self.optimizer.load_state_dict(checkpoint['opt'])
        self.scheduler.load_state_dict(checkpoint['sch'])
        self.best_acc = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']
        self.best_epoch = checkpoint['best_epoch']

        self.net.eval()

    def train_epoch(self, epoch):

        print('\nEpoch: {}'.format(epoch))
        logging.info('Epoch: {}'.format(epoch))
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tot_time = progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        logging.info('Tot: {} | Loss: {} | Acc: {}%% ({}/{})'.format(tot_time, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test_epoch(self, epoch, checkpoint='ckpt.t7'):

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        tot_time = ''
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.validloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tot_time = progress_bar(batch_idx, len(self.validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        logging.info('Tot: {} | Loss: {} | Acc: {}%% ({}/{})'.format(tot_time, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            print('Saving..')
            logging.info('Saving..')
            state = {
                'net': self.net.state_dict(),
                'cri': self.criterion.state_dict(),
                'opt': self.optimizer.state_dict(),
                'sch': self.scheduler.state_dict(),
                'max_epoch': self.max_epoch,
                'acc': self.best_acc,
                'epoch': epoch,
                'best_epoch': self.best_epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/'+checkpoint)

        # Early stop
        if epoch - self.best_epoch > self.early_stopping:
            print('Early stopping..')
            logging.info('Early stopping..')
            self.terminate = True
