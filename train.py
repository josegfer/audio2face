'''
    virtualenv: 63server pytorch
    author: Yachun Li (liyachun@outlook.com)
'''
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import os
import shutil
import time
from datetime import datetime

from dataset import BlendshapeDataset
from models import LSTMNvidiaNet

from hparams import ds, n_blendshape

import argparse

parser = argparse.ArgumentParser(description = "training")
parser.add_argument('--ckp', type = str, default = None)

args = parser.parse_args()

# gpu setting
# gpu_id = 1
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

# hyper-parameters
learning_rate = 0.0001
batch_size = 256
epochs = 500

print_freq = 20
best_loss = 10000000

# data path
dataroot = 'data'
# data_path = os.path.join(dataroot, audio2bs)
data_path = dataroot
checkpoint_path = './{}/'.format(ds)
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

def main():
    global best_loss
    model = LSTMNvidiaNet(num_blendshapes = n_blendshape)
    # print(model)
    # model = nn.DataParallel(model)

    # get data
    train_loader = torch.utils.data.DataLoader(
        BlendshapeDataset(feature_file = os.path.join(data_path, '{}_feature-lpc_train.npy'.format(ds)), 
                          target_file = os.path.join(data_path, '{}_blendshape_train.txt'.format(ds))), 
        batch_size = batch_size, shuffle = True, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(
        BlendshapeDataset(feature_file = os.path.join(data_path, '{}_feature-lpc_val.npy'.format(ds)), 
                          target_file = os.path.join(data_path, '{}_blendshape_val.txt'.format(ds))), 
        batch_size = batch_size, shuffle = False, num_workers = 0)

    # define loss and optimiser
    criterion = nn.MSELoss() #??.cuda()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.9)

    if torch.cuda.is_available():
        model = model.cuda()
    
    epoch_loaded = 0
    if args.ckp != None:
        ckp = args.ckp + '.pth.tar'
        checkpoint = torch.load(os.path.join(checkpoint_path, ckp))
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizers_state_dict'])
        epoch_loaded = checkpoint['epoch']

    # training
    print('------------\n Training begin at %s' % datetime.now())
    for epoch in range(epoch_loaded, epochs):
        start_time = time.time()

        model.train()
        train_loss = 0.
        for i, (input, target) in enumerate(train_loader):
            target = target.cuda(non_blocking = True)
            input_var = autograd.Variable(input.float()).cuda()
            target_var = autograd.Variable(target.float())

            # compute model output
            # audio_z, bs_z, output = model(input_var, target_var)
            # loss = criterion(output, target_var)
            # audio_z, bs_z, output, mu, logvar = model(input_var, target_var) # method2: loss change
            # loss = loss_function(output, target_var, mu, logvar)
            output = model(input_var)
            loss = criterion(output, target_var)

            train_loss += loss * 100 # print decimals

            # compute gradient and do the backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if i % print_freq == 0:
            #     print('Training -- epoch: {} | iteration: {}/{} | loss: {:.6f} \r'
            #             .format(epoch+1, i, len(train_loader), loss.data[0]))

        steplr.step()
        train_loss /= len(train_loader)
        # print('Glance at training   z: max/min of hidden audio(%.4f/%.4f), blendshape(%.4f/%.4f)'
        #     % (max(audio_z.data[0]), min(audio_z.data[0]), max(bs_z.data[0]), min(bs_z.data[0])))

        model.eval()
        eval_loss = 0.
        with torch.no_grad():
            for input, target in val_loader:
                target = target.cuda(non_blocking = True)
                input_var = autograd.Variable(input.float()).cuda()
                target_var = autograd.Variable(target.float())
    
                # compute output temporal?!!
                # audio_z, bs_z, output = model(input_var, target_var)
                # loss = criterion(output, target_var)
                # audio_z, bs_z, output, mu, logvar = model(input_var, target_var) # method2: loss change
                # loss = loss_function(output, target_var, mu, logvar)
                output = model(input_var)
                loss = criterion(output, target_var)
    
                eval_loss += loss * 100

        eval_loss /= len(val_loader)

        # count time of 1 epoch
        past_time = time.time() - start_time

        # print('Glance at validating z: max/min of hidden audio(%.4f/%.4f), blendshape(%.4f/%.4f)'
        #     % (max(audio_z.data[0]), min(audio_z.data[0]), max(bs_z.data[0]), min(bs_z.data[0])))

        # print('Evaluating -- epoch: {} | loss: {:.6f} \r'.format(epoch+1, eval_loss/len(val_loader)))
        print('epoch: {:03} | train_loss: {:.9f} | eval_loss: {:.9f} | {:.4f} sec/epoch \r'
            .format(epoch+1, train_loss, eval_loss, past_time))

        # save best model on val
        is_best = eval_loss < best_loss
        best_loss = min(eval_loss, best_loss)
        if is_best:
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'eval_loss': best_loss,
                }, checkpoint_path+'model_best.pth.tar')

        # save models every 100 epoch
        if (epoch+1) % 100 == 0:
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'eval_loss': eval_loss,
                }, checkpoint_path+'checkpoint-epoch'+str(epoch+1)+'.pth.tar')

    print('Training finished at %s' % datetime.now())

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, checkpoint_path+filename)
    if is_best:
        shutil.copyfile(checkpoint_path+filename, checkpoint_path+'model_best.pth.tar')

if __name__ == '__main__':
    main()
