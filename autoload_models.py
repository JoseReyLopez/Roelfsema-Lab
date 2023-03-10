from MODEL_cadena_model_inception import InceptionModel
from MODEL_cadena_model_vgg import VggModel
from MODEL_neural_model import Model
from lucent.modelzoo import inceptionv1, vgg19

import torch
import numpy as np
import os
from glob import glob





def FindLowestCadenaVgg(ROI, layer):
    assert isinstance(ROI, int) and isinstance(layer, int)
    models = glob(f'/home/jose/Desktop/lucent-things/Results_Cadena_Vgg/**/*_{ROI}_{layer}_*.pt', recursive=1)

    models2 = []; models2 = [model.split('/')[-1] for model in models]
    losses = [float(model2.split('_')[model2.split('_').index('loss')+1]) for model2 in models2]

    return models[np.argmin(losses)]


###############################################################################


def CadenaVgg(ROI = None, layer = None, autoload = True, file_name = None, eval = 1):


    from torch.cuda import is_available
    # Checking the types
    assert isinstance(ROI,   int)  or ROI   is None, 'ROI must be int or None'
    assert isinstance(layer, int)  or layer is None, 'Layer must be int or None'
    assert isinstance(autoload, bool) or autoload in [0,1]
    assert isinstance(file_name,  str)  or file_name is None, 'file_name must be str or None'
    
    neurons_per_roi = {1: 43, 2: 62, 3: 55, 4: 50, 5: 51, 7: 53, 8: 60, 9: 43, 10: 44, 11: 53, 12: 29, 13: 17, 14: 24, 15: 59, 16: 24}
    pretrained_vgg19 = vgg19(pretrained=1)
        

    if isinstance(ROI, int) or isinstance(layer, int):
        assert isinstance(ROI, int) and isinstance(layer, int) and file_name is None, 'A ROI or a Layer has been provided,'\
                                                                                       'both ROI and layer must be provided at the same time'\
                                                                                       'and file_name must NOT be provided'
        
        # Key: Roi number    Values: number of active neurons
        try:
            net = VggModel(pretrained_model = pretrained_vgg19, conv_layer = layer, num_neurons = neurons_per_roi[ROI], device = 'cuda:0' if is_available() else 'cpu')
        except:
            print('Not possible to load, most likely due to asking for a layer not accepted')
            return 0

        if autoload:
            try:
                model_path = FindLowestCadenaVgg(ROI, layer)
            except:
                print('No available model with those parameters')
                return 0

            
            net.load_state_dict(torch.load(model_path, map_location = torch.device('cuda:0' if is_available() else 'cpu')))
            
            if eval:
                net = net.to('cuda:0' if is_available() else 'cpu').eval()
            print('\n\nLoaded '+model_path.split('/')[-1])
            return net

        
    else:
        
        assert isinstance(file_name, str) and ROI is None and layer is None
        assert os.path.exists(file_name), 'The file does not exits'


        file_name_splitted = file_name.split('/')[-1]  # Checking in case the direction provided is type /home/..../Vgg..., if not, no problem anyway
        print(file_name)
        file_name_splitted = file_name_splitted.split('_')

        assert file_name_splitted[0] == 'Vgg'

        ROI   = int(file_name_splitted[2])
        layer = int(file_name_splitted[3])
        print(ROI, layer)
        net = VggModel(pretrained_model = pretrained_vgg19, conv_layer = layer, num_neurons = neurons_per_roi[ROI], device = 'cuda:0' if is_available() else 'cpu')
        
        net.load_state_dict(torch.load(file_name, map_location=torch.device('cuda:0' if is_available() else 'cpu')))

        if eval:
            net = net.to('cuda:0' if is_available() else 'cpu').eval()
        return net
    








#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#









def FindLowestCadenaInception(ROI, layer):
    assert isinstance(ROI, int) and isinstance(layer, str)
    assert layer in ['conv2d0','conv2d1','conv2d2','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b'], ''\
        'Possible options: conv2d0, conv2d1, conv2d2, mixed4a, mixed4b, mixed4c, mixed4d, mixed4e, mixed5a, mixed5b'
    
    models = glob(f'/home/jose/Desktop/lucent-things/Results_Cadena_Inception/**/*_{ROI}_{layer}_*.pt', recursive=1)

    models2 = []; models2 = [model.split('/')[-1] for model in models]
    losses = [float(model2.split('_')[model2.split('_').index('loss')+1]) for model2 in models2]

    return models[np.argmin(losses)]


###############################################################################


def CadenaInception(ROI = None, layer = None, autoload = True, file_name = None, eval = 1):

    from torch.cuda import is_available
    # Checking the types
    assert isinstance(ROI,   int)  or ROI   is None, 'ROI must be int or None'
    assert isinstance(layer, str)  or layer is None, 'Layer must be str or None'
    if layer != None:
        assert layer in ['conv2d0','conv2d1','conv2d2','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']
    assert isinstance(autoload, bool) or autoload in [0,1]
    assert isinstance(file_name,  str)  or file_name is None, 'file_name must be str or None'
    
    neurons_per_roi = {1: 43, 2: 62, 3: 55, 4: 50, 5: 51, 7: 53, 8: 60, 9: 43, 10: 44, 11: 53, 12: 29, 13: 17, 14: 24, 15: 59, 16: 24}
    pretrained_inception = inceptionv1(pretrained=1)
        

    if isinstance(ROI, int) or isinstance(layer, str):
        assert isinstance(ROI, int) and isinstance(layer, str) and file_name is None, 'A ROI or a Layer has been provided,'\
                                                                                       'ROI must be int and layer must str and provided at the same time'\
                                                                                       'and file_name must NOT be provided'
        
        # Key: Roi number    Values: number of active neurons
        try:
            net = InceptionModel(pretrained_model = pretrained_inception, layer = layer, num_neurons = neurons_per_roi[ROI], device = 'cuda:0' if is_available() else 'cpu')
        except:
            print('Not possible to load, most likely due to asking for a layer not accepted')
            return 0

        if autoload:
            try:
                model_path = FindLowestCadenaInception(ROI, layer)
                print(model_path)
            except:
                print('No available model with those parameters')
                return 0
            
            net.load_state_dict(torch.load(model_path, map_location = torch.device('cuda:0' if is_available() else 'cpu')))

            if eval:
                net = net.to('cuda:0' if is_available() else 'cpu').eval()
            print('\n\nLoaded '+model_path.split('/')[-1])
            return net

        
    else:
        
        assert isinstance(file_name, str) and ROI is None and layer is None
        assert os.path.exists(file_name), 'The file does not exits'


        file_name_splitted = file_name.split('/')[-1]  # Checking in case the direction provided is type /home/..../Vgg..., if not, no problem anyway
        print(file_name)
        file_name_splitted = file_name_splitted.split('_')

        assert file_name_splitted[0] == 'Inception'

        ROI   = int(file_name_splitted[2])
        layer = file_name_splitted[3]
        print(ROI, layer)
        net = InceptionModel(pretrained_model = pretrained_inception, layer = layer, num_neurons = neurons_per_roi[ROI], device = 'cuda:0' if is_available() else 'cpu')
        
        net.load_state_dict(torch.load(file_name, map_location=torch.device('cuda:0' if is_available() else 'cpu')))
        
        if eval:
            net = net.to('cuda:0' if is_available() else 'cpu').eval()
        return net








#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#





    

def FindLowestBashivanInception(ROI, layer):
    assert isinstance(ROI, int) and isinstance(layer, str)
    assert layer in ['conv2d0','conv2d1','conv2d2','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b'], ''\
        'Possible options: conv2d0, conv2d1, conv2d2, mixed4a, mixed4b, mixed4c, mixed4d, mixed4e, mixed5a, mixed5b'
    
    models = glob(f'/home/jose/Desktop/lucent-things/Results_Bashivan_Inception/**/*_{ROI}_{layer}_*.pt', recursive=1)

    models2 = []; models2 = [model.split('/')[-1] for model in models]
    losses = [float(model2.split('_')[model2.split('_').index('loss')+1]) for model2 in models2]

    return models[np.argmin(losses)]


###############################################################################


def BashivanInception(ROI = None, layer = None, autoload = True, file_name = None, eval = 1):

    from torch.cuda import is_available
    import collections
    
    # Checking the types
    assert isinstance(ROI,   int)  or ROI   is None, 'ROI must be int or None'
    assert isinstance(layer, str)  or layer is None, 'Layer must be str or None'
    if layer != None:
        assert layer in ['conv2d0','conv2d1','conv2d2','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']
    assert isinstance(autoload, bool) or autoload in [0,1]
    assert isinstance(file_name,  str)  or file_name is None, 'file_name must be str or None'
    
    neurons_per_roi = {1: 43, 2: 62, 3: 55, 4: 50, 5: 51, 7: 53, 8: 60, 9: 43, 10: 44, 11: 53, 12: 29, 13: 17, 14: 24, 15: 59, 16: 24}
    pretrained_inception = inceptionv1(pretrained=1)
        

    if isinstance(ROI, int) or isinstance(layer, str):
        assert isinstance(ROI, int) and isinstance(layer, str) and file_name is None, 'A ROI or a Layer has been provided,'\
                                                                                       'ROI must be int and layer must str and provided at the same time'\
                                                                                       'and file_name must NOT be provided'
        
        # Key: Roi number    Values: number of active neurons
        try:
            net = Model(pretrained_model = pretrained_inception, layer = layer, n_neurons = neurons_per_roi[ROI], device = 'cuda:0' if is_available() else 'cpu')
        except:
            print('Not possible to load, most likely due to asking for a layer not accepted')
            return 0

        if autoload:
            try:
                model_path = FindLowestBashivanInception(ROI, layer)
                print(model_path)
            except:
                print('No available model with those parameters')
                return 0
            
            # This network had problem matching the keys, so I have to make a new OrderedDict with the keys changing
            # from 'module.w_s' to 'w_s', that's what I have to do here...

            loaded_model = torch.load(model_path, map_location = torch.device('cuda:0' if is_available() else 'cpu'))
            # loaded_model is an OrderedDict
            loaded_model_new_keys = collections.OrderedDict((k.split('module.', 1)[1],v) for k,v in loaded_model.items())
            net.load_state_dict(loaded_model_new_keys)

            if eval:
                net = net.to('cuda:0' if is_available() else 'cpu').eval()
            print('\n\nLoaded '+model_path.split('/')[-1])
            return net

        
    else:
        
        assert isinstance(file_name, str) and ROI is None and layer is None
        assert os.path.exists(file_name), 'The file does not exits'


        file_name_splitted = file_name.split('/')[-1]  # Checking in case the direction provided is type /home/..../Vgg..., if not, no problem anyway
        print(file_name)
        file_name_splitted = file_name_splitted.split('_')

        assert file_name_splitted[1] == 'Inception'

        ROI   = int(file_name_splitted[2])
        layer = file_name_splitted[3]

        print(ROI, layer)
        
        net = Model(pretrained_model = pretrained_inception, layer = layer, n_neurons = neurons_per_roi[ROI], device = 'cuda:0' if is_available() else 'cpu')
        
        loaded_model = torch.load(file_name, map_location=torch.device('cuda:0' if is_available() else 'cpu'))
        loaded_model_new_keys = collections.OrderedDict((k.split('module.', 1)[1],v) for k,v in loaded_model.items())
        net.load_state_dict(loaded_model_new_keys)
        
        if eval:
            net = net.to('cuda:0' if is_available() else 'cpu').eval()
        return net






#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#







def FindLowestBashivanVgg(ROI, layer):
    assert isinstance(ROI, int) and isinstance(layer, int)

    models = glob(f'/home/jose/Desktop/lucent-things/Results_Bashivan_Vgg/**/*_{ROI}_{layer}_*.pt', recursive=1)

    models2 = []; models2 = [model.split('/')[-1] for model in models]
    losses = [float(model2.split('_')[model2.split('_').index('loss')+1]) for model2 in models2]

    return models[np.argmin(losses)]



###############################################################################


def BashivanVgg(ROI = None, layer = None, autoload = True, file_name = None, eval = 1):

    from torch.cuda import is_available
    
    # Checking the types
    assert isinstance(ROI,   int)  or ROI   is None, 'ROI must be int or None'
    assert isinstance(layer, int)  or layer is None, 'Layer must be str or None'
    assert isinstance(autoload, bool) or autoload in [0,1]
    assert isinstance(file_name,  str)  or file_name is None, 'file_name must be str or None'
    
    neurons_per_roi = {1: 43, 2: 62, 3: 55, 4: 50, 5: 51, 7: 53, 8: 60, 9: 43, 10: 44, 11: 53, 12: 29, 13: 17, 14: 24, 15: 59, 16: 24}
    pretrained_vgg19 = vgg19(pretrained=1)
        

    if isinstance(ROI, int) or isinstance(layer, int):
        assert isinstance(ROI, int) and isinstance(layer, int) and file_name is None, 'A ROI or a Layer has been provided,'\
                                                                                       'ROI and layer must be int and provided at the same time'\
                                                                                       'and file_name must NOT be provided'
        
        # Key: Roi number    Values: number of active neurons
        try:
            net = Model(pretrained_model = pretrained_vgg19, layer = 'features.'+str(layer), n_neurons = neurons_per_roi[ROI], device = 'cuda:0' if is_available() else 'cpu')
        except:
            print('Not possible to load, most likely due to asking for a layer not accepted')
            return 0

        if autoload:
            try:
                model_path = FindLowestBashivanVgg(ROI, layer)
                print(model_path)
            except:
                print('No available model with those parameters')
                return 0
            
            net.load_state_dict(torch.load(model_path, map_location = torch.device('cuda:0' if is_available() else 'cpu')))

            if eval:
                net = net.to('cuda:0' if is_available() else 'cpu').eval()
            print('\n\nLoaded '+model_path.split('/')[-1])
            return net

        
    else:
        
        assert isinstance(file_name, str) and ROI is None and layer is None
        assert os.path.exists(file_name), 'The file does not exits'


        file_name_splitted = file_name.split('/')[-1]  # Checking in case the direction provided is type /home/..../Vgg..., if not, no problem anyway
        print(file_name)
        file_name_splitted = file_name_splitted.split('_')

        assert file_name_splitted[1] == 'Vgg'

        ROI   = int(file_name_splitted[2])
        layer = file_name_splitted[3]
        print(ROI, layer)
        net = Model(pretrained_model = pretrained_vgg19, layer = 'features.'+str(layer), n_neurons = neurons_per_roi[ROI], device = 'cuda:0' if is_available() else 'cpu')
        
        net.load_state_dict(torch.load(file_name, map_location=torch.device('cuda:0' if is_available() else 'cpu')))

        if eval:
            net = net.to('cuda:0' if is_available() else 'cpu').eval()
        return net





########################################################################




def layer_minimization_per_roi(model, CNN):
    
    '''
    A model ['Cadena', 'Bashivan'] and a pretrained cnn ['Vgg', 'Inception'] are provided
    and the function returns which is the layer (for Vgg [0,2,5,7,10,12,14...], for Inception
    ['conv2d0', 'conv2d1'...])
    The associated roi is encoded in the possition of the list. If no model was trained for 
    a given roi, that position will contain a None.
    '''


    assert model in ['Cadena', 'Bashivan']
    assert CNN in ['Vgg', 'Inception']

    ########


    neurons_per_roi = {1: 43, 2: 62, 3: 55, 4: 50, 5: 51, 7: 53, 8: 60, 9: 43, 10: 44, 11: 53, 12: 29, 13: 17, 14: 24, 15: 59, 16: 24}

    vgg_layers = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]
    inception_layers  = ['conv2d0', 'conv2d1', 'conv2d2', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b']
    rois = list(range(1,17))


    ########



    if CNN == 'Vgg':
        layers = vgg_layers
    if CNN == 'Inception':
        layers = inception_layers

    max_layer_per_roi = []

    for roi in rois:

        active_layers = []
        losses_per_layer = []
        for layer in layers:
            #print(roi, layer)
            model_name = None
            if model == 'Cadena' and CNN == 'Vgg':
                try:
                    model_name = FindLowestCadenaVgg(roi, layer)
                except:
                    pass
            if model == 'Cadena' and CNN == 'Inception':
                try:
                    model_name = FindLowestCadenaInception(roi, layer)
                except:
                    pass
            if model == 'Bashivan' and CNN == 'Vgg':
                try:
                    model_name = FindLowestBashivanVgg(roi, layer)
                except:
                    pass
            if model == 'Bashivan' and CNN == 'Inception':
                try:
                    model_name = FindLowestBashivanInception(roi, layer)
                except:
                    pass
            
            if model_name != None:
                #print(roi, layer)
                splitted_name = model_name.split('/')[-1].rsplit('.', 1)[0].split('_')
                losses_per_layer.append(float(splitted_name[splitted_name.index('loss')+1]))
                active_layers.append(layer)

        #print(roi, losses_per_layer, active_layers)
        if len(losses_per_layer)>0:
            max_layer_per_roi.append(active_layers[np.argmin(losses_per_layer)])
        else:
            max_layer_per_roi.append(None)
            
    return max_layer_per_roi