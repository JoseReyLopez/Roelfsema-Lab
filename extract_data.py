import os
import h5py
import pickle
import numpy as np

import torch
from torchvision import datasets, transforms

import sys
roi_name=sys.argv[1]

if os.name == 'posix':
    filename = '/home/jose/Desktop/Data/data_THINGS_array'+ roi_name +'_v1.pkl'
    data_path = '/home/jose/Desktop/Data/THINGS_exportMUA_array'+ roi_name + '.mat'
    imgs_path = '/home/jose/Desktop/Data/THINGS_imgs/train/'
    val_imgs_path = '/home/jose/Desktop/Data/THINGS_imgs/val/'
if os.name == 'nt':
    filename = 'E:/Jose/Data/data_THINGS_array'+ roi_name +'_v1.pkl'
    data_path = 'E:/Jose/Data/THINGS_exportMUA_array'+ roi_name + '.mat'
    imgs_path = 'E:/Jose/Data/THINGS_imgs/train/'
    val_imgs_path = 'E:/Jose/Data/THINGS_imgs/val/'

data_dict = {}
f = h5py.File(data_path,'r')
for k, v in f.items():
    data_dict[k] = np.array(v)
    
val_data = data_dict['test_MUA'].squeeze()
train_data = data_dict['train_MUA'].squeeze()
n_neurons = train_data.shape[1]
del data_dict

transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
dataset = datasets.ImageFolder(imgs_path, transform=transform)
dataset_val = datasets.ImageFolder(val_imgs_path, transform=transform)

debug = 0
if debug:
    idxs_path = 'Data/THINGS_normMUA.mat'
    data_dict = {}
    f = h5py.File(idxs_path,'r')
    for k, v in f.items():
        data_dict[k] = np.array(v)
    idx_temp = data_dict['train_idx'].astype(int).squeeze() - 1
    temp_subset = torch.utils.data.Subset(dataset, idx_temp)
    dataloader = torch.utils.data.DataLoader(temp_subset, batch_size=train_data.shape[0], shuffle=False)
else:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_data.shape[0], shuffle=False)
    
img_data, junk = next(iter(dataloader))

val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=train_data.shape[0], shuffle=False)
val_img_data, junk = next(iter(val_dataloader))

output = {'img_data':img_data, 'val_img_data':val_img_data,
          'train_data':train_data, 'val_data':val_data}
 
f = open(filename,"wb")
pickle.dump(output,f,protocol=4)
f.close()
print('Done')