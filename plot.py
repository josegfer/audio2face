#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:51:27 2022

@author: jose
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from hparams import ds

# read
data_path = 'data'
result_path = './output/'

target_file = os.path.join(data_path, '{}_blendshape_test.txt'.format(ds))
predicted_file = os.path.join(result_path, '{}_hat.txt'.format(ds))

y = np.loadtxt(target_file)
y_hat = np.loadtxt(predicted_file)

# plot map
figure, axis = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6));

idx = np.arange(0, 44*30, 1, dtype = int) # first 44 seconds in 30 fps

axis[0].imshow(y[idx, :], aspect = 'auto', vmin = 0, vmax = 1, cmap = 'gray');
axis[0].set_title('y');
axis[1].imshow(y_hat[idx, :], aspect = 'auto', vmin = 0, vmax = 1, cmap = 'gray');
axis[1].set_title('y_hat');
figure.savefig(os.path.join(result_path, '{}_map.png'.format(ds)))

# plot signal
figure, axis = plt.subplots(nrows = 2, ncols = 2, figsize = (32, 12));

axis[0, 0].plot(y[idx, 2], label = 'y');
axis[0, 0].plot(y_hat[idx, 2], label = 'y_hat');
axis[0, 0].set_title('[:, 2]');
axis[0, 0].legend();
axis[0, 1].plot(y[idx, 3], label = 'y');
axis[0, 1].plot(y_hat[idx, 3], label = 'y_hat');
axis[0, 1].set_title('[:, 3]');
axis[0, 1].legend();

axis[1, 0].plot(y[idx, 30], label = 'y');
axis[1, 0].plot(y_hat[idx, 30], label = 'y_hat');
axis[1, 0].set_title('[:, 30]');
axis[1, 0].legend();
axis[1, 1].plot(y[idx, 32], label = 'y');
axis[1, 1].plot(y_hat[idx, 32], label = 'y_hat');
axis[1, 1].set_title('[:, 32]');
axis[1, 1].legend();

figure.savefig(os.path.join(result_path, '{}_signal.png'.format(ds)))
