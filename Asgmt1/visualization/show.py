#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : show
Author          : Charles Young
Python Version  : Python 3.6.1
Date            : 2018-10-23
'''
print(__doc__)

import numpy as np
import pandas as pd
import os
import sys
import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')

sys.path.insert(0,'..')

from util import mnist_helper
from util import transform

if __name__ == '__main__':
	# Get dataset and split into train and test
	train_x, train_y, test_x, test_y = mnist_helper.get_dataset('../data/')

	fig = plt.figure(figsize = (10, 10))

	# Display 25 raw digits.
	for i in range(25):
		ax = fig.add_subplot(5, 5, i + 1)
		ax.set_axis_off()
		ax.imshow(train_x.values[i].reshape(28, 28))
		ax.set_title('Label \'{}\''.format(train_y[i]))

	if not os.path.exists(os.path.join('..', 'figs')):
		os.makedirs(os.path.join('..', 'figs'))
	plt.savefig(os.path.join('..', 'figs', 'show_raw_digits.svg'), bbox_inches = 'tight', format = 'svg')

	train_x = transform.deskew(train_x.values[:25])

	# Display 25 deskewed digits.
	for i in range(25):
		ax = fig.add_subplot(5, 5, i + 1)
		ax.set_axis_off()
		ax.imshow(train_x[i].reshape(28, 28))
		ax.set_title('Label \'{}\''.format(train_y[i]))

	if not os.path.exists(os.path.join('..', 'figs')):
		os.makedirs(os.path.join('..', 'figs'))
	plt.savefig(os.path.join('..', 'figs', 'show_deskewed_digits.svg'), bbox_inches = 'tight', format = 'svg')
