print('start')

import os
import pickle
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np

import torch
from torchvision import datasets, transforms
import torchvision.models as models 
from torch import sigmoid
from torch.cuda.amp import GradScaler, autocast

from lucent.util import set_seed
from lucent.modelzoo import inceptionv1, util
from helper import smoothing_laplacian_loss, sparsity_loss
from MODEL_neural_model import Model
from directory_tools import create_folder, renumber_subfolders, clean_up_folders, delete_all_but_lowest_validation_file

import sys
import argparse


torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('-r','--roi',   required=True, type=int)
parser.add_argument('-l','--layer', required=True, type=str)   # I should input 0  or 2  or 5 etc, not conv2d2 etc...

parser.add_argument('-lr', '--learningrate',    type=float, default = 1e-2)
parser.add_argument('-smo', '--smoothweights',  type=float, default = 0.01)
parser.add_argument('-spw', '--sparseweights',  type=float, default = 1e-8)
parser.add_argument('-wdc', '--weightdecays',   type=float, default = 0.001)

args = parser.parse_args()

roi_name = str(args.roi)
print(roi_name)

layer      = args.layer
layers = [layer]

# Paolo's original layers    layer_options =  ['conv2d2', 'mixed4d', 'mixed4a']
#                    ✅           ✅       ✅         ✅         ✅         ✅          ✅        ✅         ✅         ✅  
layer_options  = ['conv2d0', 'conv2d1', 'conv2d2', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b']
assert layer in layer_options, 'Layer options are conv2d0, conv2d1, conv2d2, mixed4a, mixed4b, mixed4c, mixed4d, mixed4e, mixed5a, mixed5b'



# hyperparameters
grid_search = True
seed = 0
nb_epochs = 1500
save_epochs = 100
grid_epochs = 120
grid_save_epochs = grid_epochs
batch_size = 100
lr_decay_gamma = 1/3
lr_decay_step = 3*save_epochs
backbone = 'inception_v1'


'''
if int(roi_name) < 9: #V1
    layers = ['conv2d2']
    learning_rates = [1e-2,1e-3,1e-4]
    smooth_weights = [0.05,0.1,0.15,0.2]
    weight_decays = [0.01,0.05,0.1,0.2]
elif int(roi_name) > 12: #IT
    layers = ['mixed4d']
    learning_rates = [0.05,1e-2,1e-3]
    smooth_weights = [0.01,0.05,0.1,0.15]
    weight_decays = [0.001,0.005,0.01,0.05]
else: #V4
    layers = ['mixed4a']
    learning_rates = [1e-2,1e-3,1e-4]
    smooth_weights = [0.001,0.005,0.01,0.05]
    weight_decays = [0.0001,0.0005,0.001,0.005]
'''




#learning_rates = [1e-2] #,1e-3,1e-4]
#smooth_weights = [0.01]#,0.05,0.1,0.15]
#weight_decays  = [0.001]#,0.005,0.01,0.05]   # For Adam
#sparse_weights = [1e-8]

learning_rates = [args.learningrate]
smooth_weights = [args.smoothweights]
weight_decays  = [args.weightdecays]
sparse_weights = [args.sparseweights]


data_filename      = 'E:/Jose/Data/data_THINGS_array'+ roi_name +'_v1.pkl'

current_datetime = time.strftime("%Y-%m-%d_%H_%M_%S", time.gmtime())

if os.name == 'nt':
    #data_filename = '/media/stijn/2bb74e85-3681-4561-88b7-abd98482de61/paolo/Data/data_THINGS_array'+ roi_name +'_v1.pkl'
    grid_filename = 'E:/Jose/snapshots/grid_search_array'+ roi_name +'.pkl'
    snapshot_path = f'E:/Jose/snapshots/array'+ roi_name +'_neural_model.pt'
    loss_plot_path = f'E:/Jose/training_data/training_loss_classifier_{current_datetime}.png'

if os.name == 'posix':

    os.chdir('/home/jose/Desktop/lucent-things/Results_Bashivan_Inception')

    data_filename  = '/home/jose/Desktop/Data/data_THINGS_array'+roi_name+'_v1.pkl'
    grid_filename  = '/home/jose/Desktop/snapshots/grid_search_array'+roi_name+'.pkl'
#    snapshot_path  = '/home/jose/Desktop/snapshots/array'+roi_name+'_neural_model_'+date+'.pt'
    results_save   = '/home/jose/Desktop/lucent-things/Results_Bashivan_Inception'
    loss_plot_path = '/home/jose/Desktop/training_data/training_loss_classifier_{current_datetime}.png'


clean_up_folders()
renumber_subfolders(extension='Inception')
create_folder(args.roi, args.layer, extension='Inception')


load_snapshot = False
GPU = torch.cuda.is_available()
print(GPU)
if GPU:
    torch.cuda.set_device(0)
    
snapshot_pattern = 'E:/Jose/snapshots/neural_model_{backbone}_{layer}.pt'

# load data
f = open(data_filename,"rb")
cc = pickle.load(f)

train_data = cc['train_data']
val_data = cc['val_data']
img_data = cc['img_data']
val_img_data = cc['val_img_data']
n_neurons = train_data.shape[1]

del cc

######
# Grid search:
######
iter1_done = False
if grid_search:
    params = []
    val_corrs = []
    for layer in layers:
        print('======================')
        print('Backbone: ' + layer)
        for learning_rate in learning_rates:
            for smooth_weight in smooth_weights:
                for sparse_weight in sparse_weights:
                    for weight_decay in weight_decays:
                        
                        set_seed(seed)
                        if iter1_done:
                            del pretrained_model
                            del net
                            del criterion
                            del scaler
                            del optimizer
                            del scheduler
                            
                        iter1_done = True
                        # model, wrapped in DataParallel and moved to GPU
                        pretrained_model = inceptionv1(pretrained=True)
                        if GPU:
                            net = Model(pretrained_model,layer,n_neurons,torch.device("cuda:0" if GPU else "cpu"))
                            if load_snapshot:
                                net.load_state_dict(torch.load(
                                    snapshot_path,
                                    map_location=lambda storage, loc: storage
                                ))
                                print('Loaded snap ' + snapshot_path)
                            else:
                                net.initialize()
                            #net = torch.nn.DataParallel(net)
                            net = net.cuda()
                        else:
                            net = Model(pretrained_model,layer,n_neurons,torch.device("cuda:0" if GPU else "cpu"))
                            if load_snapshot:
                                net.load_state_dict(torch.load(
                                    snapshot_path,
                                    map_location=lambda storage, loc: storage
                                ))
                                print('Loaded snap ' + snapshot_path)
                            else:
                                net.initialize()
                                print('Initialized using Xavier')
                            net = torch.nn.DataParallel(net)
                            print("Training on CPU")

                        # loss function
                        criterion = torch.nn.MSELoss()
                        
                        scaler = GradScaler()

                        # optimizer and lr scheduler
                        optimizer = torch.optim.Adam(
                            #[net.module.w_s,net.module.w_f],
                            [net.w_s,net.w_f],
                            lr=learning_rate,
                            weight_decay=weight_decay)
                        scheduler = torch.optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=lr_decay_step,
                            gamma=lr_decay_gamma)

                        cum_time = 0.0
                        cum_time_data = 0.0
                        cum_loss = 0.0
                        optimizer.step()
                        for epoch in range(grid_epochs):
                            # adjust learning rate
                            scheduler.step()
                            torch.cuda.empty_cache()
                            # get the inputs & wrap them in tensor
                            batch_idx = np.random.choice(np.linspace(0,train_data.shape[0]-1,train_data.shape[0]), 
                                                         size=batch_size,replace=False).astype('int')
                            if GPU:
                                neural_batch = torch.tensor(train_data[batch_idx,:]).cuda()
                                val_neural_data = torch.tensor(val_data).cuda()
                                img_batch = img_data[batch_idx,:].cuda()
                            else:
                                neural_batch = torch.tensor(train_data[batch_idx,:])
                                val_neural_data = torch.tensor(val_data)
                                img_batch = img_data[batch_idx,:]

                            # forward + backward + optimize
                            tic = time.time()
                            with autocast():
                                optimizer.zero_grad()
                                outputs = net(img_batch).squeeze()
                                #loss = criterion(outputs, neural_batch.float()) + smoothing_laplacian_loss(net.module.w_s, 
                                #                                                        torch.device("cuda:0" if GPU else "cpu"), 
                                #                                                        weight=smooth_weight) \
                                #                                                + sparse_weight * torch.norm(net.module.w_f,1)
                                loss = criterion(outputs, neural_batch.float()) + smoothing_laplacian_loss(net.w_s, 
                                                                                        torch.device("cuda:0" if GPU else "cpu"), 
                                                                                        weight=smooth_weight) \
                                                                                + sparse_weight * torch.norm(net.w_f,1)                                

                            #loss.backward()
                            #optimizer.step()
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            toc = time.time()
                                    
                            cum_time += toc - tic
                            cum_loss += float(loss)
                            
                            del neural_batch 
                            del val_neural_data 
                            del img_batch 
                            del outputs

                            # output & test
                            if epoch % grid_save_epochs == grid_save_epochs - 1:
                                with torch.no_grad(): 
                                    torch.cuda.empty_cache()
                                    net.to('cpu').eval()
                                    val_outputs = net(val_img_data).squeeze().numpy()
                                    #val_loss = criterion(val_outputs, val_neural_data)
                                    corrs = []
                                    for n in range(val_outputs.shape[1]):
                                        corrs.append(np.corrcoef(val_outputs[:,n],val_data[:,n])[1,0])
                                    val_corr = np.median(corrs)

                                    #tic_test = time.time()
                                    # print and plot time / loss
                                    print('======')
                                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                                    no_epoch = epoch / (grid_save_epochs - 1)
                                    #mean_time = cum_time / float(grid_save_epochs)
                                    mean_loss = cum_loss / float(grid_save_epochs)
                                    #if mean_loss.is_cuda:
                                    #    mean_loss = mean_loss.detach().cpu()
                                    cum_time = 0.0
                                    cum_loss = 0.0
                                
                                del val_outputs

                        params.append([layer,learning_rate,smooth_weight,sparse_weight,weight_decay])
                        val_corrs.append(val_corr)
                        #print('======================')
                        print(f'learning rate: {learning_rate}')
                        print(f'smooth weight: {smooth_weight}')
                        print(f'sparse weight: {sparse_weight}')
                        print(f'weight decay: {weight_decay}')
                        print(f'Validation corr: {val_corr:.3f}')
                        #print('======')

    # extract winning params
    val_corrs = np.array(val_corrs)
    layer = params[np.where(val_corrs==val_corrs.max())[0][0].astype('int')][0]
    learning_rate = params[np.where(val_corrs==val_corrs.max())[0][0].astype('int')][1]
    smooth_weight = params[np.where(val_corrs==val_corrs.max())[0][0].astype('int')][2]
    sparse_weight = params[np.where(val_corrs==val_corrs.max())[0][0].astype('int')][3]
    weight_decay = params[np.where(val_corrs==val_corrs.max())[0][0].astype('int')][4]

    # print winning params
    print('======================')
    print('Best backbone is: ' + layer)
    print('Best learning rate is: ' + str(learning_rate))
    print('Best smooth weight is: ' + str(smooth_weight))
    print('Best sparse weight is: ' + str(sparse_weight))
    print('Best weight decay is: ' + str(weight_decay))
    
    # print winning params
    output = {'val_corrs':val_corrs, 'params':params}
    f = open(grid_filename,"wb")
    pickle.dump(output,f)
    f.close()
else:   
    if int(roi_name) < 9: #V1
        layer = 'conv2d2'
        learning_rate = 0.001
        smooth_weight  = 0.05
        sparse_weight  = 1e-08
        weight_decay  = 0.01
    elif int(roi_name) > 12: #IT (most likely)
        layer = 'mixed4d'
        learning_rate = 0.01
        smooth_weight  = 0.05
        sparse_weight  = 1e-08
        weight_decay  = 0.005
    else: #V4
        layer = 'mixed4a'
        learning_rate = 0.001
        smooth_weight  = 0.001
        sparse_weight  = 1e-08
        weight_decay  = 0.001


######
# Final training!!
######

# model, wrapped in DataParallel and moved to GPU
set_seed(seed)
pretrained_model = inceptionv1(pretrained=True)
if GPU:
    net = Model(pretrained_model,layer,n_neurons,torch.device("cuda:0" if GPU else "cpu"))
    if load_snapshot:
        net.load_state_dict(torch.load(
            snapshot_path,
            map_location=lambda storage, loc: storage
        ))
        print('Loaded snap ' + snapshot_path)
    else:
        net.initialize()
        print('Initialized using Xavier')
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    print("Training on {} GPU's".format(torch.cuda.device_count()))
else:
    net = Model(pretrained_model,layer,n_neurons,torch.device("cuda:0" if GPU else "cpu"))
    if load_snapshot:
        net.load_state_dict(torch.load(
            snapshot_path,
            map_location=lambda storage, loc: storage
        ))
        print('Loaded snap ' + snapshot_path)
    else:
        net.initialize()
        print('Initialized using Xavier')
    net = torch.nn.DataParallel(net)
    print("Training on CPU")

# loss function
criterion = torch.nn.MSELoss()

# optimizer and lr scheduler
optimizer = torch.optim.Adam(
    [net.module.w_s,net.module.w_f],
    lr=learning_rate,
    weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=lr_decay_step,
    gamma=lr_decay_gamma)

# figure for loss function
fig = plt.figure()
axis = fig.add_subplot(111)
axis.set_xlabel('epoch')
axis.set_ylabel('loss')
axis.set_yscale('log')
plt_line, = axis.plot([], [])

cum_time = 0.0
cum_time_data = 0.0
cum_loss = 0.0
optimizer.step()
for epoch in range(nb_epochs):
    # adjust learning rate
    scheduler.step()

    # get the inputs & wrap them in tensor
    batch_idx = np.random.choice(np.linspace(0,train_data.shape[0]-1,train_data.shape[0]), 
                                 size=batch_size,replace=False).astype('int')
    if GPU:
        neural_batch = torch.tensor(train_data[batch_idx,:]).cuda()
        val_neural_data = torch.tensor(val_data).cuda()
        img_batch = img_data[batch_idx,:].cuda()
    else:
        neural_batch = torch.tensor(train_data[batch_idx,:])
        val_neural_data = torch.tensor(val_data)
        img_batch = img_data[batch_idx,:]

    # forward + backward + optimize
    tic = time.time()
    optimizer.zero_grad()
    outputs = net(img_batch).squeeze()
    loss = criterion(outputs, neural_batch.float()) + smoothing_laplacian_loss(net.module.w_s, 
                                                                      torch.device("cuda:0" if GPU else "cpu"), 
                                                                      weight=smooth_weight) \
                                                    + sparse_weight * torch.norm(net.module.w_f,1)

    loss.backward()
    optimizer.step()
    toc = time.time()

    cum_time += toc - tic
    cum_loss += loss.data.cpu()

    # output & test
    if epoch % save_epochs == save_epochs - 1:

        val_outputs = net(val_img_data).squeeze()
        val_loss = criterion(val_outputs, val_neural_data)
        corrs = []
        for n in range(val_outputs.shape[1]):
            corrs.append(np.corrcoef(val_outputs[:,n].cpu().detach().numpy(),val_data[:,n])[1,0])
        val_corr = np.median(corrs)

        tic_test = time.time()
        # print and plot time / loss
        print('======================')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        no_epoch = epoch / (save_epochs - 1)
        mean_time = cum_time / float(save_epochs)
        mean_loss = cum_loss / float(save_epochs)
        if mean_loss.is_cuda:
            mean_loss = mean_loss.data.cpu()
        cum_time = 0.0
        cum_loss = 0.0
        print(f'epoch {np.int(epoch)}/{nb_epochs} mean time: {mean_time:.3f}s')
        print(f'epoch {np.int(epoch)}/{nb_epochs} mean loss: {mean_loss:.3f}')
        print(f'epoch {np.int(no_epoch)} validation loss: {val_loss:.3f}')
        print(f'epoch {np.int(no_epoch)} validation corr: {val_corr:.3f}')
        plt_line.set_xdata(np.append(plt_line.get_xdata(), no_epoch))
        plt_line.set_ydata(np.append(plt_line.get_ydata(), mean_loss))
        axis.relim()
        axis.autoscale_view()

        fig.savefig(loss_plot_path)

        print('======================')
        print('Test time: ', time.time()-tic_test)


        torch.save(net.state_dict(), f'Bashivan_Inception_{args.roi}_{args.layer}_epoch_{int(no_epoch)}_loss_{val_loss}_smoothweight_{args.smoothweights}_sparseweight_{args.sparseweights}_weightdecay_{args.weightdecays}.pt')
        delete_all_but_lowest_validation_file()

# save the weights, we're done!
#os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
#torch.save(net.module.state_dict(), snapshot_path)
