#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : pca_visual
Author          : Charles Young
Python Version  : Python 3.6.1
Date            : 2018-10-24
'''
print(__doc__)

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('agg')

cm = np.array([[976, 0, 0, 1, 0, 0, 2, 0, 1, 0],
 [0, 128, 2, 2, 1, 0, 1, 0, 1, 0],
 [2, 1, 1017, 1, 1, 1, 1, 3, 5, 0],
 [0, 0, 1, 1000, 0, 3, 0, 2, 2, 2],
 [0, 0, 3, 0, 971, 0, 3, 0, 0, 5],
 [1, 0, 0, 6, 0, 884, 1, 0, 0, 0],
 [4, 3, 0, 1, 4, 2, 942, 0, 2, 0],
 [0, 0, 2, 0, 0, 1, 0, 1024, 0, 1],
 [2, 0, 2, 3, 1, 1, 0, 1, 963, 1],
 [1, 3, 1, 4, 8, 3, 1, 5, 1, 982]])

if __name__ == '__main__':

	fig = plt.figure(figsize = (10, 10))

	plt.figure(figsize=(9,9))
	sns.heatmap(cm, annot = True, linewidths = .5, square = True, cmap = 'Pastel1')
	plt.ylabel('Actual label');
	plt.xlabel('Predicted label')
	all_sample_title = 'Accuracy Score: {0}'.format(0.9887)
	plt.title(all_sample_title, size = 15)

	if not os.path.exists(os.path.join('..', 'figs')):
		os.makedirs(os.path.join('..', 'figs'))
	plt.savefig(os.path.join('..', 'figs', 'cm_visual.svg'), bbox_inches = 'tight', format = 'svg')
