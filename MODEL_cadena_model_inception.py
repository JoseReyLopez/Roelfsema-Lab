import torch
from torch import nn, Tensor
import torch.nn.functional as F
from helper import makeGaussian, FeatureExtractor, fix_parameters, load_sta
from lucent.modelzoo import vgg19, inceptionv1, util
import matplotlib.pyplot as plt
import numpy as np

class InceptionModel(nn.Module):

    def __init__(self, pretrained_model, layer = 'conv2d2', num_neurons = 43, device = None):
        super(InceptionModel, self).__init__()

        self.features_to_extract = ['conv2d2', 'mixed4d', 'mixed4a']
        self.layer = layer
        print('Layer:  ', self.layer, type(self.layer))
        print('FTE:         ', self.features_to_extract)
        print('conv in FTE', self.layer in self.features_to_extract)
        print('\n\n')

        self.inception_pretrained = pretrained_model
        self.ann = fix_parameters(self.inception_pretrained)
        self.feature_extractor = FeatureExtractor(self.ann, layers = [self.layer])

        dummy_input  = torch.ones(1, 3, 224, 224)
        dummy_output = self.feature_extractor(dummy_input)
        #print(dummy_output.keys(), dummy_output[list(dummy_output.keys())[0]].shape)
        dummy_output_shape = list(dummy_output[list(dummy_output.keys())[0]].shape)
        w_shape = dummy_output_shape[1:] + [num_neurons]
        print(w_shape)

        self.biases = nn.Parameter(torch.randn(1, num_neurons, device = device))
        self.w      = nn.Parameter(torch.randn(w_shape,        device = device))
        
        self.BN1    = nn.BatchNorm2d(num_features=w_shape[0])
        self.output = nn.Identity()

    #def extract_features(self, x):
    #    self.vgg_features = self.feature_extractor(x)
    #    print(self.vgg_features.keys())
    #    self.feature_number = list(self.vgg_features.keys())[0].split('.')[-1]
    #    return self.vgg_features[list(self.vgg_features.keys())[0]]

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x[self.layer]
        x = F.relu(self.BN1(x))
        x = torch.tensordot(x, self.w, [[1,2,3],[0,1,2]])
        x = x + self.biases
        return self.output(x)

    def initialize(self):
        nn.init.xavier_normal_(self.biases)
        nn.init.xavier_normal_(self.w)
        
        
    def shape(self):
        return self.w
    def mod_shape(self):
        return self.w_shape

if 0:
    incept_pretrained = inceptionv1(pretrained = True)
    incept = InceptionModel(incept_pretrained)
    sw     = incept.shape()

