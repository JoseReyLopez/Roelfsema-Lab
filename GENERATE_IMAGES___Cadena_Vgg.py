from glob import glob
import numpy as np
import os
from autoload_models import *

import torch
from lucent.modelzoo import vgg19, util
from lucent.optvis import render, param, transform, objectives

import matplotlib.pyplot as plt
import itertools
import os

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-r','--roi',   required=True, type=int)
parser.add_argument('-l','--layer', required=True, type=int)  
args = parser.parse_args()


neurons_per_roi = {1: 43, 2: 62, 3: 55, 4: 50, 5: 51, 7: 53, 8: 60, 9: 43, 10: 44, 11: 53, 12: 29, 13: 17, 14: 24, 15: 59, 16: 24}


all_transforms = [
    transform.jitter(10),
]

if not os.path.exists('Results_no_transform'):
    os.mkdir('Results_no_transform')

os.chdir('Results_no_transform')


if not os.path.exists('Cadena_Vgg'):
    os.mkdir('Cadena_Vgg')

os.chdir('Cadena_Vgg')



net = CadenaVgg(args.roi, args.layer)

param_f = lambda: param.image(128, fft=True, batch = 5, decorrelate=True)
cppn_opt = lambda params: torch.optim.Adam(params, 1e-2, weight_decay=1e-4)
obj = objectives.channel('output', np.array(list(range(neurons_per_roi[args.roi])))) # Still need to add the correlation stuff here
img = render.render_vis(net, obj, param_f, cppn_opt, transforms=all_transforms, show_inline=False)

np.save(f'Cadena_Vgg_roi_{args.roi}_layer_{args.layer}', img[0])
f, ax = plt.subplots(1, 5, figsize = (45, 10))
for i in range(5):
    ax[i].imshow(img[0][i, :])
    ax[i].set_axis_off()

plt.savefig(f'Cadena_Vgg_Img_roi_{args.roi}_layer_{args.layer}.png', dpi = 150)
    
