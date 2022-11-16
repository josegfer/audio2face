#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:39:24 2022

@author: jose
"""

import numpy as np
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from hparams import ds, eta, n

data_path = 'data'

feature_file = os.path.join('..', data_path, '{}_feature-lpc.npy'.format(ds))
target_file = os.path.join('..', data_path, '{}_blendshape.txt'.format(ds))
metric_file = os.path.join('..', data_path, '{}_indices.npy'.format(ds))

# read
X = np.load(feature_file)
y = np.loadtxt(target_file)

# resample
metric = np.load(metric_file)
N = len(metric)
sortido = np.argsort(metric)[-int(N * eta):]

X_resample = X[sortido, : , :]
y_resample = y[sortido, :]

# concatenate
for i in range(n):
    X = np.concatenate((X, X_resample), axis = 0)
    y = np.concatenate((y, y_resample), axis = 0)

# write
np.save(os.path.join('..', data_path, 'resample_{}_feature-lpc.npy'.format(ds)), X)
np.savetxt(os.path.join('..', data_path, 'resample_{}_blendshape.txt'.format(ds)), y, fmt='%.8f')
