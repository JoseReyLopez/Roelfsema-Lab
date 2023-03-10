import os
import pickle
import sys
import time
import glob
import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np

import torch
#from torchvision import datasets, transforms
import torchvision.models as models 
from torch import sigmoid

from lucent.util import set_seed
from lucent.modelzoo import vgg19, util
from helper import smoothing_laplacian_loss, sparsity_loss
from cadena_model import VggModel

############     My scripts for my model    ############
from GPU_selection import GPU_selection
from make_directory import create_folder, clean_up_folders, renumber_subfolders
from plot_training_summary import plot_training_summary
from clean_models import delete_all_but_lowest_validation_file



import sys
roi_name   = sys.argv[1]
layer      = sys.argv[2]

if len(sys.argv) == 4:
    device_gpu = sys.argv[3]
    device_gpu = int(sys.argv[3])

#roi_name = '9'
#layer = 0
print('\nROI:  ', roi_name, '\nLayer:  ', layer)


################  Making directory and changing moving into it
if os.name == 'posix':   #When it runs at donders
    os.chdir('/home/jose/Desktop/lucent-things/Results')
if os.name == 'nt':      #When it runs at NINs
    os.chdir('E:/Jose/Results')

clean_up_folders()
renumber_subfolders()
create_folder(int(roi_name), layer)
#################################################


### Loss functions

def w_sum(weight):
    return torch.sum(torch.abs(weight))

def smoothness_loss(weight):
    Laplacian = torch.tensor([[[0,-1,0],[-1,-4,-1],[0,-1,0]]])
    Laplacian_holder =torch.zeros([n_neurons,n_neurons,3,3])
    for i in range(n_neurons):
        for j in range(n_neurons):
            Laplacian_holder[i,j,:,:] = Laplacian
    weight_reshaped = weight.reshape([weight.shape[0], weight.shape[3], weight.shape[1], weight.shape[2]])
    smoothness_loss = torch.sqrt(torch.sum(torch.square(torch.nn.functional.conv2d(weight_reshaped.cuda(), Laplacian_holder.cuda()))))
    return smoothness_loss

def sparsity_loss(weight):
    weight_reshaped = weight.reshape([weight.shape[0], weight.shape[3], weight.shape[1], weight.shape[2]])
    sparsity_loss = torch.sum(torch.square(weight_reshaped*2), axis = [2,3,1,0]) 
    return sparsity_loss




# hyperparameters
grid_search = True
seed = 0
nb_epochs = 1000
save_epochs = 100
grid_epochs = 120
grid_save_epochs = grid_epochs
batch_size = 100
lr_decay_gamma = 1/3
lr_decay_step = 3*save_epochs


vgg_pretrained = vgg19(pretrained = True)


if os.name == 'posix':   #When it runs at donders
    data_filename  = '/home/jose/Desktop/Data/data_THINGS_array'+ roi_name +'_v1.pkl'
if os.name == 'nt':      #When it runs at NINs
    data_filename  = 'E:/Jose/Data/data_THINGS_array'+ roi_name +'_v1.pkl'

print(data_filename)
GPU = torch.cuda.is_available()
if GPU:
    if len(sys.argv) == 4:
        gpu_to_use = device_gpu

    else:
        gpu_to_use = GPU_selection()

    #gpu_to_use = 0
    print('\n\nGPU to use: '+str(gpu_to_use)+'\n\n')
    device = 'cuda:' + str(gpu_to_use)
    print(device)



# load data
f = open(data_filename,"rb")
cc = pickle.load(f)

train_data = cc['train_data']
val_data = cc['val_data']
img_data = cc['img_data']
val_img_data = cc['val_img_data']
n_neurons = train_data.shape[1]
print('\n\nData Loaded\n\n')        

# layer options
# 0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34
# "cuda:"+str(gpu_to_use) if GPU else "cpu"
net = VggModel(pretrained_model=vgg_pretrained, conv_layer=layer, num_neurons=n_neurons, device = torch.device(device))
net.initialize()
net = net.cuda()

#if not(os.path.exists('raw_model_'+str(roi_name)+'_'+str(layer)+'.pt')):
    #torch.save(net.state_dict(), 'raw_model.pt')
#    torch.save(net, 'raw_model_'+str(roi_name)+'_'+str(layer)+'.pt')

print('\n\n\nDone!!!')

# How to save a model
#torch.save(net.state_dict(), 'E:/Jose/models/rm90.pt') 
#
####
# How to load a model
# from lucent.modelzoo import vgg19, util
# from cadena_model import VggModel
# vgg_pretrained = vgg19(pretrained = True)
# net = VggModel(pretrained_model=vgg_pretrained, conv_layer=layer, num_neurons=n_neurons, device = torch.device("cuda:"+str(gpu_to_use) if GPU else "cpu"))
# net.cuda()
# net.load_state_dict(torch.load('E:/Jose/models/rm90.pt'))
#


##########
lr = 1e-6
epochs = 20

train_dataset_size = img_data.shape[0]
val_dataset_size   = val_img_data.shape[0]

batch_size = 50

smooth_weight = 0.1
sparse_weight = 1e-2

##########

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam([net.w, net.biases], lr = lr)


#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 70, gamma = 0.97)

lr_list = []
loss_list = []
val_loss_list = []

col_names =  ['epoch', 'epochs', 'batch', 'batchs', 'loss train', 'validation loss', 'learning rate', 'l1 loss', 'smooth loss', 'sparsity loss', 'time']
data_matrix = np.zeros((1, len(col_names)))

time0 = time.time()

for epoch in range(epochs):
    for batch in range(int(train_dataset_size/batch_size)):
        torch.cuda.empty_cache()
        batch_idx         = np.random.choice([i for i in range(22248)], size = batch_size, replace = False)
        neural_batch      = torch.tensor(train_data[batch_idx, :]).cuda()
        img_batch         = torch.tensor(img_data[batch_idx, :]).cuda()

        val_img_batch     = torch.tensor(val_img_data).cuda()
        val_neural_batch  = torch.tensor(val_data).cuda()

        outputs = net(img_batch).squeeze()
        loss = criterion(outputs, neural_batch.float())\
            + torch.sum(torch.abs(net.w))\
            + smooth_weight * smoothness_loss(net.w)\
            + sparse_weight * sparsity_loss(net.w)
        
        loss_train        = criterion(outputs, neural_batch.float())
        l1_loss_str       = torch.round(w_sum(net.w), decimals = 4).item()
        smooth_loss_str   = torch.round(smoothness_loss(net.w), decimals = 4).item()
        sparsity_loss_str = torch.round(sparsity_loss(net.w),decimals = 4).item()
        
        with torch.no_grad():
                    val_loss_last = torch.nn.functional.mse_loss(net(val_img_batch), val_neural_batch).item()

        #loss = criterion(outputs, neural_batch.float()) + smoothing_laplacian_loss(net.w, torch.device("cuda:"+str(gpu_to_use) if GPU else "cpu"),  weight=smooth_weight) + sparse_weight * torch.norm(net.w,1) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print('\n\n\n===========================')
        #print(val_img_data.shape)
        #print(net(val_img_batch).shape)
        #print(val_neural_batch.shape)
        #print(criterion(net(val_img_batch), val_neural_batch))
        #print('===========================\n\n\n')

        


        if len(loss_list)>0 and np.min(loss_list)>loss.item():
            marker = '----> '    # Min train loss reduced
            if len(val_loss_list)>0 and np.min(val_loss_list)>val_loss_last:
                marker = marker + '* ' # min train and val loss reduced
            else:
                marker = marker + '  ' # Just train loss
        elif len(val_loss_list)>0 and np.min(val_loss_list)>val_loss_last:
                marker = '*****   '   #  min val loss reduced
        else:
            marker = '        '   # Nothing


        if loss.item()<0.25 and np.min(loss_list)>loss.item():
            pass
            #torch.save(net, 'model_minimum_'+str(roi_name)+'_'+str(layer)+'.pt')

        if val_loss_last<.9 and val_loss_last<np.min(val_loss_list):
            pass
            #if os.name == 'nt':
            #    torch.save(net.state_dict(), 'E:/Jose/models/model_'+str(roi_name)+'_'+str(layer)+'_'+str(int(time.time()))+'_min_val_loss__'+str(val_loss_last)+'.pt')
            #if os.name == 'posix':
            #    torch.save(net.state_dict(), '/home/jose/Desktop/models/model_'+str(roi_name)+'_'+str(layer)+'_'+str(int(time.time()))+'_min_val_loss__'+str(val_loss_last)+'.pt')
        

        loss_list.append(loss_train.item())
        val_loss_list.append(val_loss_last)
        lr_list.append(scheduler.get_last_lr()[0])
        #print(f'{marker} Epoch:{epoch}/{epochs}   I:{batch}/{int(train_dataset_size/batch_size)}   Train:{np.round(loss.item(),4):.3f}   Val:{val_loss_list[-1]:.3f}   Min T:{np.min(loss_list):.3f}   Min V:{np.min(val_loss_list):.3f}   LR:{scheduler.get_last_lr()[0]:.3e}')#optimizer.param_groups[0]['lr'])
        time1 = time.time()
        print(f'{marker} Epoch:{epoch}/{epochs}   B:{batch}/{int(train_dataset_size/batch_size)}   Train:{np.round(loss_train.item(),4):.3f}   Val:{val_loss_list[-1]:.5f}   Min T:{np.min(loss_list):.3f}   Min V:{np.min(val_loss_list):.5f}   LR:{scheduler.get_last_lr()[0]:.3e}  L1:{l1_loss_str:.4f}  Smooth: {smooth_loss_str:.4f}  Sparsity: {sparsity_loss_str:.4f}  Time:{time1-time0:.2f}')#optimizer.param_groups[0]['lr'])
        scheduler.step()
        # ep  of total   batch of total                               loss train                          val_loss                  lr                     l1             smooth_loss        sparsity
        data_matrix = np.vstack((data_matrix, np.array([epoch,epochs,  batch,int(train_dataset_size/batch_size),  np.round(loss_train.item(),4),  val_loss_list[-1],   scheduler.get_last_lr()[0],  l1_loss_str,   smooth_loss_str,  sparsity_loss_str, np.round(time1-time0, 1)])))

    np.save('training_data_'+str(roi_name)+'_'+str(layer), data_matrix)


    #### Epoch data saving, model for that epoch, plotting losses
    if os.name == 'posix':
        torch.save(net.state_dict(), 'model_'+str(roi_name)+'_'+str(layer)+'_epoch_'+str(epoch)+'_loss_'+str(val_loss_list[-1])+'.pt')
    if os.name == 'nt':
            torch.save(net.state_dict(), 'model_'+str(roi_name)+'_'+str(layer)+'_epoch_'+str(epoch)+'_loss_'+str(val_loss_list[-1])+'.pt')
    delete_all_but_lowest_validation_file()


    ###################### Plotting losses etc
    plot_training_summary('training_data_'+str(roi_name)+'_'+str(layer)+'.npy')