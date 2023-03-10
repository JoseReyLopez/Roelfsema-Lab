import torch
from torch import nn, Tensor
import torch.nn.functional as F
from helper import makeGaussian, FeatureExtractor, fix_parameters, load_sta
from lucent.modelzoo import vgg19, util
import matplotlib.pyplot as plt
import numpy as np

class VggModel(nn.Module):

    def __init__(self, pretrained_model, conv_layer = 0, num_neurons = 43, device = None):
        super(VggModel, self).__init__()

        
        self.features_to_extract = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]
        self.conv_layer = conv_layer
        print('conv_layer:  ', self.conv_layer, type(self.conv_layer))
        print('FTE:         ', self.features_to_extract)
        print('conv in FTE',   int(self.conv_layer) in self.features_to_extract)
        print('\n\n')

        self.vgg_pretrained = pretrained_model
        self.ann = fix_parameters(self.vgg_pretrained)
        self.feature_extractor = FeatureExtractor(self.ann, layers = ['features.'+str(self.conv_layer)])

        dummy_input  = torch.ones(1, 3, 224, 224)
        dummy_output = self.feature_extractor(dummy_input)    #print(dummy_output.keys(), dummy_output[list(dummy_output.keys())[0]].shape)
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
        x = x['features.' + str(self.conv_layer)]
        x = F.relu(self.BN1(x))
        x = torch.tensordot(x, self.w, [[1,2,3],[0,1,2]])
        x = x + self.biases
        return self.output(x)

    def initialize(self):
        nn.init.xavier_normal_(self.biases)
        nn.init.xavier_normal_(self.w)

test = 0

if test:

    
    vgg_pretrained = vgg19(pretrained = True)

    vgge = VggModel(vgg_pretrained)


    im_tensor = torch.zeros(4, 3, 224, 224)
    for i in range(1,5):
        im_test = torch.Tensor(plt.imread('/home/jose/Desktop/Data/THINGS_imgs/val/test/0000'+str(i)+'.bmp'))
        im_test = np.transpose(im_test, (2, 0, 1))
        im_test = im_test[ : , 500-224:500, 100:100+224]
        print(i, '-', im_test.shape)
        im_tensor[i-1, :, :, :] = torch.Tensor(im_test)

    vgge.forward(im_tensor).shape






