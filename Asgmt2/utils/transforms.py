#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : transform
Author          : Charles Young
Python Version  : Python 3.6.2
Date            : 2018-12-01
'''

import random
import math
import numpy as np
import torch

class RandomErasing(object):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.

    Parameters
    ----------
    p : float
        Erasing p
    sl : float
        Minimum erasing S ratio
    sh : float
        Maximum erasing S ratio
    r1 : float
        Minimum erasing aspect ratio
    mean : 3x1 array containing elements value from 0-255
        Pixel values to replace the erasing S with

    """    

    def __init__(self, p = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.p = p
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.p:
            return img

        for attempt in range(100):
            S = img.size()[1] * img.size()[2]
       
            Se = random.uniform(self.sl, self.sh) * S
            re = random.uniform(self.r1, 1/self.r1)

            He = int(round(math.sqrt(Se * re)))
            We = int(round(math.sqrt(Se / re)))

            if We < img.size()[2] and He < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - He)
                y1 = random.randint(0, img.size()[2] - We)
                if img.size()[0] == 3:
                    img[0, x1:x1+He, y1:y1+We] = self.mean[0]
                    img[1, x1:x1+He, y1:y1+We] = self.mean[1]
                    img[2, x1:x1+He, y1:y1+We] = self.mean[2]
                else:
                    img[0, x1:x1+He, y1:y1+We] = self.mean[0]
                return img

        return img
