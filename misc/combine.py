import numpy as np
import os
import sys
import json
from tqdm import tqdm
import pandas as pd

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from hparams import ds, win_size, K, n_blendshape

dataroot = '../data'
feature_path = os.path.join(dataroot, 'feature-lpc/')
target_path = os.path.join(dataroot, 'blendshape/')

test_ind = pd.read_csv('../data/test_ind.csv', header = None).squeeze('columns').tolist() # audio2face/data/blendshape$ ls rule* > ../test_ind.csv
val_ind = pd.read_csv('../data/val_ind.csv', header = None).squeeze('columns').tolist() # audio2face/data/blendshape$ ls rule* > ../val_ind.csv

def combine(feature_path, target_path, conjunto = 'train'):
    sets = {'train', 'val', 'test'}
    assert conjunto in sets
    
    feature_files = sorted(os.listdir(feature_path))
    # print('feature: ', feature_files)

    blendshape_files = sorted(os.listdir(target_path))
    # print('bs:      ', blendshape_files)

    # feature = np.array([], dtype = np.float64).reshape(0, win_size, K) # [frames x win_size x K] from lpc
    feature = []
    # feature_combine_file = ds + '_' + feature_path.split('/')[-2] + '.npy'
    feature_combine_file = '{}_{}_{}.npy'.format(ds, feature_path.split('/')[-2], conjunto)

    # blendshape = np.array([], dtype = np.float64).reshape(0, n_blendshape) # [frames x n_blendshape]
    blendshape = []
    blendshape_combine_file = ds + '_' + target_path.split('/')[-2] + '.txt'
    blendshape_combine_file = '{}_{}_{}.txt'.format(ds, target_path.split('/')[-2], conjunto)

    for i in tqdm(range(len(feature_files))):
        # skip test files
        if conjunto == 'train':
            if blendshape_files[i] in test_ind or blendshape_files[i] in val_ind:
                continue
        elif conjunto == 'val':
            if blendshape_files[i] in test_ind or blendshape_files[i] not in val_ind:
                continue
        else:
            if blendshape_files[i] not in test_ind or blendshape_files[i] in val_ind:
                continue

        feature_temp = np.load(feature_path+feature_files[i])
        # feature = np.concatenate((feature, feature_temp), 0)
        feature.append(feature_temp)

        # blendshape is shorter, need cut
        blendshape_temp = loadjson(target_path + blendshape_files[i])
        blendshape_temp = cut(feature_temp, blendshape_temp)
        # blendshape = np.concatenate((blendshape, blendshape_temp), 0)
        blendshape.append(blendshape_temp)

        # print(i, blendshape_files[i], feature.shape, blendshape.shape)

    feature = np.concatenate(feature, axis = 0)
    blendshape = np.concatenate(blendshape, axis = 0)
    
    np.save(os.path.join(dataroot, feature_combine_file), feature)
    np.savetxt(os.path.join(dataroot, blendshape_combine_file), blendshape, fmt='%.8f')

def cut(wav_feature, blendshape_target):
    n_audioframe, n_videoframe = len(wav_feature), len(blendshape_target)
    # print('--------\n', 'Current dataset -- n_audioframe: {}, n_videoframe:{}'.format(n_audioframe, n_videoframe))
    assert n_videoframe - n_audioframe == 32
    start_videoframe = 16
    blendshape_target = blendshape_target[start_videoframe : start_videoframe + n_audioframe]

    return blendshape_target

def loadjson(blendshape_file):
    f = open(blendshape_file)
    data = json.load(f)
    f.close()
    return np.array(data['weightMat'])

def main():
    combine(feature_path, target_path, conjunto = 'train')
    combine(feature_path, target_path, conjunto = 'val')
    combine(feature_path, target_path, conjunto = 'test')

if __name__ == '__main__':
    main()
