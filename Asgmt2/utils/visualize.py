#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : visualize
Author          : Charles Young
Python Version  : Python 3.6.2
Date            : 2018-12-01
'''

import os
import sys
import re
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def transform_logfile(file):
    datetime_pattern = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}')
    tottime_pattern = re.compile(r'[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}')
    print('Process {}..'.format(file))
    sp = file.split('_')
    result = {
    'model': sp[1], 
    'time': sp[2] + '_' +sp[3].split('.')[0],
    'best_acc': (0, 0), # (acc, epoch)
    'tot_time': 0,
    'acc': {'train': [], 'test': []},
    'loss': {'train': [], 'test': []},
    }
    with open(os.path.join('../logs', file)) as reader:
        text = reader.read()
        text = re.sub('\n', ' ', text)                    # rm all enter
        text = text[text.find(':Epoch'):]                 # rm the structure log
        text = re.sub(datetime_pattern, '', text)         # rm all time strip

        # extract info of total
        info = re.findall(r':Time(.+?)/300\)', text)[0]
        info = info.strip().split()
        result['tot_time'] = info[2]
        result['best_acc'] = (float(info[6]), int(info[7][1:]))

        # extract info of each epoch
        info = re.findall(r':Epoch: (.+?)/10000\)', text)  # find all info of each epoch
        info = [group.strip().split() for group in info]   # split info of each epoch to get detailed info
        result['loss']['train'] = list(map(float, [info_epoch[5] for info_epoch in info]))
        result['loss']['test'] = list(map(float, [info_epoch[14] for info_epoch in info]))
        result['acc']['train'] = list(map(float, [info_epoch[8][:-2] for info_epoch in info]))
        result['acc']['test'] = list(map(float, [info_epoch[17][:-2] for info_epoch in info]))
    print("Process done. Saving {}...".format(file))
    if not os.path.isdir(os.path.join('../results/dics', sp[0])):
        os.makedirs(os.path.join('../results/dics', sp[0]))
    file = 'dic_' + sp[1] + '_acc' + str(result['best_acc'][0]) + '.json'
    with open(os.path.join('../results/dics', sp[0], file), "w") as writer:
        json.dump(result, writer)
    return result

def plot_overlap(logger, field, name):
    numbers = logger[field][name]
    x = np.arange(len(numbers))
    if field == 'acc':
        plt.plot(x, 100.0-np.asarray(numbers, dtype='float'))
    else:
        plt.plot(x, np.asarray(numbers))
    return [logger['model'] + ' (' + name + ')']

def multiplot(files, field, name, prefix):
    plt.figure()
    plt.plot()
    legend_text = []
    for file in files:
        if file.startswith(prefix):  # only process the logs that the corresponding trains were stopping.
            logger = transform_logfile(file)
            legend_text += plot_overlap(logger, field, name)
    plt.legend(legend_text, loc=0)
    if field == 'acc':
        plt.ylabel('error (%)')
    else:
        plt.ylabel(field)
    plt.xlabel('epoch')
    plt.grid(True)
                    
def savefig(fname, dpi, prefix):
    if not os.path.isdir(os.path.join('../results/figs/', prefix[3:-1])):
        os.makedirs(os.path.join('../results/figs/', prefix[3:-1]))
    plt.savefig(os.path.join('../results/figs/', prefix[3:-1], fname), dpi=dpi)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type = str, default = 'acc', choices = ['acc', 'loss'], nargs = '?', 
                        help = "Field to plot. Default is 'acc'")
    parser.add_argument('--name', type = str, default = 'test', choices = ['test', 'train'], nargs = '?', 
                        help = "Name to plot. Default is 'test'")
    parser.add_argument('--prefix', type = str, default = 'logD_', choices = ['logD_', 'logDR_'], nargs = '?', 
                        help = "Prefix of filename of the log to be extracted. Default is 'logD'")
    parser.add_argument('--dpi', type = int, default = 150, nargs = '?',
                        help = "DPI of the plot. Default is 150")
    args = parser.parse_args()

    for root, dirs, files in os.walk('../logs/'):
        nb_files = len(files)
        print('Tot number of files: {}'.format(nb_files))        
        multiplot(files=files, field=args.field, name=args.name, prefix=args.prefix)
        savefig(args.field + '_' + args.name + '.svg', args.dpi, prefix=args.prefix)
