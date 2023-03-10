import os
import pickle
import sys
import time
import glob
import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse
import torch
#from torchvision import datasets, transforms
import torchvision.models as models 
from torch import sigmoid

from lucent.util import set_seed
from lucent.modelzoo import inceptionv1, util
from helper import smoothing_laplacian_loss, sparsity_loss
from MODEL_cadena_model_inception import InceptionModel

############     My scripts for my model    ############
from GPU_selection import GPU_selection
from directory_tools import create_folder, clean_up_folders, renumber_subfolders, delete_all_but_lowest_validation_file
from plot_training_summary import plot_training_summary

import sys

########################################################


torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('-r','--roi',   required=True, type=int)
parser.add_argument('-l','--layer', required=True, type=str)
parser.add_argument('-e','--epoch', type=int, default=10)

parser.add_argument('-g', '--gpu', type=int, default=0)

parser.add_argument('-spa', '--sparsity',       type=float, default = 10e-3)
parser.add_argument('-smo', '--smoothness',     type=float, default = 10e-2)
parser.add_argument('-gsp', '--groupsparsity', type=float, default = 10e-4)

args = parser.parse_args()

print(f'Regularization hyperparamameters:   -Sparsity: ', args.sparsity, '   -smoothness: ', args.smoothness ,'   -group sparsity', args.groupsparsity)


roi_name   = str(args.roi)
layer      = args.layer

# Paolo's original layers    layer_options =  ['conv2d2', 'mixed4d', 'mixed4a']
#                    ✅           ✅       ✅         ✅         ✅         ✅          ✅        ✅         ✅         ✅  
layer_options  = ['conv2d0', 'conv2d1', 'conv2d2', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b']
assert layer in layer_options, 'Layer options are conv2d0, conv2d1, conv2d2, mixed4a, mixed4b, mixed4c, mixed4d, mixed4e, mixed5a, mixed5b'

## ERROR TO SOLVE:  ValueError: zero-size array to reduction operation minimum which has no identity


if len(sys.argv) == 4:
    device_gpu = sys.argv[3]
    device_gpu = int(sys.argv[3])

#roi_name = '9'
#layer = 0
print('\nROI:  ', roi_name, '\nLayer:  ', layer)


################  Making directory and changing moving into it
if os.name == 'posix':   #When it runs at donders
    os.chdir('/home/jose/Desktop/lucent-things/Results_Cadena_Inception')
if os.name == 'nt':      #When it runs at NINs
    os.chdir('E:/Jose/Results_Cadena_Inception')

clean_up_folders()
renumber_subfolders()
create_folder(int(roi_name), layer, extension='Inception')
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


inception_pretrained = inceptionv1(pretrained = True)

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
# "cuda:"+str(gpu_to_use) if GPU else "cpu"
net = InceptionModel(pretrained_model=inception_pretrained, layer=layer, num_neurons=n_neurons, device = torch.device(device))
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
lr = 1e-5
epochs = args.epoch

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
        neural_batch      = torch.tensor(train_data[batch_idx, :]).clone().cuda()
        img_batch         = torch.tensor(img_data[batch_idx, :]).clone().cuda()

        val_img_batch     = torch.tensor(val_img_data).clone().cuda()
        val_neural_batch  = torch.tensor(val_data).clone().cuda()

        outputs = net(img_batch).squeeze()
        loss = criterion(outputs, neural_batch.float())\
            + 0.001*torch.sum(torch.abs(net.w))\
            + 0.1*smooth_weight * smoothness_loss(net.w)\
            + 0.01*sparse_weight * sparsity_loss(net.w)
        
        loss_train        = criterion(outputs, neural_batch.float())
        l1_loss_str       = torch.round(w_sum(net.w), decimals = 4).item()
        smooth_loss_str   = torch.round(smoothness_loss(net.w), decimals = 4).item()
        sparsity_loss_str = torch.round(sparsity_loss(net.w),decimals = 4).item()
        

        #loss = criterion(outputs, neural_batch.float()) + smoothing_laplacian_loss(net.w, torch.device("cuda:"+str(gpu_to_use) if GPU else "cpu"),  weight=smooth_weight) + sparse_weight * torch.norm(net.w,1) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        marker = '        '   # Nothing



        loss_list.append(loss_train.item())
        lr_list.append(scheduler.get_last_lr()[0])
        #print(f'{marker} Epoch:{epoch}/{epochs}   I:{batch}/{int(train_dataset_size/batch_size)}   Train:{np.round(loss.item(),4):.3f}   Val:{val_loss_list[-1]:.3f}   Min T:{np.min(loss_list):.3f}   Min V:{np.min(val_loss_list):.3f}   LR:{scheduler.get_last_lr()[0]:.3e}')#optimizer.param_groups[0]['lr'])
        time1 = time.time()
        if epoch!=0:
            print(f'{marker} Epoch:{epoch}/{epochs}   B:{batch}/{int(train_dataset_size/batch_size)}   Train:{np.round(loss_train.item(),4):.3f}   Val:{val_loss_list[-1]:.5f}   Min T:{np.min(loss_list):.3f}   Min V:{np.min(val_loss_list):.5f}   LR:{scheduler.get_last_lr()[0]:.3e}  L1:{l1_loss_str:.4f}  Smooth: {smooth_loss_str:.4f}  Sparsity: {sparsity_loss_str:.4f}  Time:{time1-time0:.2f}')#optimizer.param_groups[0]['lr'])
            scheduler.step()
            # ep  of total   batch of total                               loss train                          val_loss                  lr                     l1             smooth_loss        sparsity
            data_matrix = np.vstack((data_matrix, np.array([epoch,epochs,  batch,int(train_dataset_size/batch_size),  np.round(loss_train.item(),4),  val_loss_list[-1],   scheduler.get_last_lr()[0],  l1_loss_str,   smooth_loss_str,  sparsity_loss_str, np.round(time1-time0, 1)])))

        else:
            print(f'{marker} Epoch:{epoch}/{epochs}   B:{batch}/{int(train_dataset_size/batch_size)}   Train:{np.round(loss_train.item(),4):.3f}   Min T:{np.min(loss_list):.3f}  LR:{scheduler.get_last_lr()[0]:.3e}  L1:{l1_loss_str:.4f}  Smooth: {smooth_loss_str:.4f}  Sparsity: {sparsity_loss_str:.4f}  Time:{time1-time0:.2f}')#optimizer.param_groups[0]['lr'])
            scheduler.step()
            # ep  of total   batch of total                               loss train                                         lr                     l1             smooth_loss        sparsity
            data_matrix = np.vstack((data_matrix, np.array([epoch,epochs,  batch,int(train_dataset_size/batch_size),  np.round(loss_train.item(),4),  1000,  scheduler.get_last_lr()[0],  l1_loss_str,   smooth_loss_str,  sparsity_loss_str, np.round(time1-time0, 1)])))



    np.save('inception_training_data_'+str(roi_name)+'_'+str(layer), data_matrix)

    with torch.no_grad():
        val_loss_last = torch.nn.functional.mse_loss(net(val_img_batch), val_neural_batch).item()
        val_loss_list.append(val_loss_last)

    print('\n\n'+' '*38+'######################################################################################')
    print(    ' '*38+'################                    Epoch finished                    ################')
    print(    ' '*38+'################                                                      ################')
    print(    ' '*38+f'################                 Validation Loss:  {np.round(val_loss_last, 5)}            ################')
    print(    ' '*38+'######################################################################################\n\n')

    #### Epoch data saving, model for that epoch, plotting losses
    model_name = 'Inception_model_'+str(roi_name)+'_'+str(layer)+'_epoch_'+str(epoch)+'_loss_'+str(val_loss_list[-1])
    model_name = f'{model_name}_sparsity_{args.sparsity}_smoothness_{args.smoothness}_groupsparsity_{args.groupsparsity}.pt'
    torch.save(net.state_dict(), model_name)
    delete_all_but_lowest_validation_file()

    

    ###################### Plotting losses etc
    #plot_training_summary('inception_training_data_'+str(roi_name)+'_'+str(layer)+'.npy')