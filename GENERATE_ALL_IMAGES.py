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

import gc



neurons_per_roi = {1: 43, 2: 62, 3: 55, 4: 50, 5: 51, 7: 53, 8: 60, 9: 43, 10: 44, 11: 53, 12: 29, 13: 17, 14: 24, 15: 59, 16: 24}

vgg_layers = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]
inception_layers  = ['conv2d0', 'conv2d1', 'conv2d2', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b']
rois = list(range(1,17))



########  Cadena Vgg  ###########

#for roi, vgg_layer in itertools.product(rois, vgg_layers):
#    os.system(f'python GENERATE_IMAGES___Cadena_Vgg.py -r {roi} -l {vgg_layer}')

 

########  Cadena Inception  ###########

#for roi, inception_layer in itertools.product(rois, inception_layers):
#    os.system(f'python GENERATE_IMAGES___Cadena_Inception.py -r {roi} -l {inception_layer}')

        

########  Bashivan Inception  ###########

for roi, inception_layer in itertools.product(rois[::-1], inception_layers[::-1]):
    if inception_layer in ['conv2d1', 'conv2d2']:
        os.system(f'python GENERATE_IMAGES___Bashivan_Inception.py -r {roi} -l {inception_layer}')

    else:
        pass
   

########  Bashivan Vgg  ###########

#for roi, vgg_layer in itertools.product(rois, vgg_layers):
#    os.system(f'python GENERATE_IMAGES___Bashivan_Vgg.py -r {roi} -l {vgg_layer}')
