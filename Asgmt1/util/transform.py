#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : baseline
Author          : Charles Young
Python Version  : Python 3.6.1
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-10-06
'''

import cv2  
import numpy as np  


def affine_transform(ims, lbs):
	expanded_ims = [im for im in ims]
	expanded_lbs = [lb for lb in lbs]

	A = np.float32([[0,0],[0,0]])
	B_ls = [np.float32([1,0]), np.float32([0,1]), np.float32([-1,0]), np.float32([0,-1])
	, np.float32([1,1]), np.float32([-1,1]), np.float32([-1,-1]), np.float32([1,-1])]
	M_ls = [np.hstack((A,B.reshape((2,-1)))) for B in B_ls]

	for idx in range(len(ims)):
		for M in M_ls:
			expanded_ims.append(cv2.warpAffine(ims[idx], M, ims[idx].shape))
			expanded_lbs.append(lbs[idx])

	return np.array(expanded_ims).reshape((len(expanded_ims), ims[0].shape[0] * ims[0].shape[1])), np.array(expanded_lbs).flatten()   

