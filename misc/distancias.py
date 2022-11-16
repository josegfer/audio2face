
import numpy as np
# from numba import jit
from tqdm import tqdm
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from hparams import ds, threshold_x, threshold_y

data_path = 'data'

feature_file = os.path.join('..', data_path, '{}_feature-lpc.npy'.format(ds))
target_file = os.path.join('..', data_path, '{}_blendshape.txt'.format(ds))

X = np.load(feature_file)
y = np.loadtxt(target_file)

N = X.shape[0]

indices = np.zeros(N)
for i0, x0 in tqdm(enumerate(X)):
  problematicos = 0
  for i, x in enumerate(X):
    dx = np.linalg.norm(x0 - x)
    dy = np.linalg.norm(y[i0, :] - y[i, :])
    if (dx < threshold_x) and (dy > threshold_y):
      problematicos += 1
  indice = problematicos/N
  indices[i0] = indice

np.save(os.path.join('..', data_path, '{}_indices.npy'.format(ds)), indices)
