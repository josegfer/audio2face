import torch
import torch.autograd as autograd

import os
import time
import numpy as np
import argparse
from scipy.signal import savgol_filter

from dataset import BlendshapeDataset
from models import LSTMNvidiaNet

from hparams import ds, n_blendshape

# options
parser = argparse.ArgumentParser(description="PyTorch testing of LSTM")
parser.add_argument('ckp', type=str)
parser.add_argument('--smooth', type=bool, default=False)
parser.add_argument('--pad', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--net', type=str, default='lstm')

args = parser.parse_args()

# parameters
batch_size = 100

# data path
dataroot = 'data'
data_path = dataroot
# data_path = './data/test/'
checkpoint_path = './{}/'.format(ds)
result_path = './output/'
if not os.path.isdir(result_path): os.mkdir(result_path)

result_file = '{}_hat.txt'

if args.epoch != None:
    ckp = 'checkpoint-epoch'+str(args.epoch)+'.pth.tar'
    result_file = str(args.epoch)+'-'+result_file
else:
    ckp = args.ckp+'.pth.tar'

def pad_blendshape(blendshape):
    return np.pad(blendshape, [(16, 16), (0, 0)], mode='constant', constant_values=0.0)


model = LSTMNvidiaNet(num_blendshapes = n_blendshape)

# restore checkpoint model
print("=> loading checkpoint '{}'".format(ckp))
checkpoint = torch.load(os.path.join(checkpoint_path, ckp))
print("model epoch {} loss: {}".format(checkpoint['epoch'], checkpoint['eval_loss']))

model.load_state_dict(checkpoint['state_dict'])

# load data
val_loader = torch.utils.data.DataLoader(
    BlendshapeDataset(feature_file = os.path.join(data_path, '{}_feature-lpc_test.npy'.format(ds)), 
                      target_file = os.path.join(data_path, '{}_blendshape_test.txt'.format(ds))), 
    batch_size = batch_size, shuffle = False, num_workers = 0)

if torch.cuda.is_available():
    model = model.cuda()

# run test features
model.eval()

start_time = time.time()
with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking = True)
        input_var = autograd.Variable(input.float()).cuda()
        target_var = autograd.Variable(target.float())
    
        # compute output
        output = model(input_var)
    
        if i == 0:
            output_cat = output.data
        else:
            output_cat = torch.cat((output_cat, output.data), 0)
        # print(type(output_cat.cpu().numpy()), output_cat.cpu().numpy().shape)

# convert back *100
output_cat = output_cat.cpu().numpy()*100.0

if args.smooth:
    #smooth3--savgol_filter
    win = 9; polyorder = 3
    for i in range(n_blendshape):
        power = output_cat[:,i]
        power_smooth = savgol_filter(power, win, polyorder, mode='nearest')
        output_cat[:, i] = power_smooth
    result_file = 'smooth-' + result_file

# padding to the same frames as input wav
if args.pad:
    output_cat = pad_blendshape(output_cat)
    result_file = 'pad-' + result_file

# count time for testing
past_time = time.time() - start_time
print("Test finished in {:.4f} sec! Saved in {}".format(past_time, result_file))

with open(os.path.join(result_path, result_file), 'wb') as f:
    np.savetxt(f, output_cat, fmt='%.6f')
