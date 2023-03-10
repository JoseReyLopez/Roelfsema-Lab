def correlations(model:str, verbose=0):

    import torch
    from lucent.modelzoo import vgg19, util
    from MODEL_cadena_model_vgg import VggModel
    from MODEL_cadena_model_inception import InceptionModel

    vgg_pretrained = vgg19(pretrained = True)

    gpu_to_use = 0

    neurons_per_roi = {1: 43,   # Key: Roi number    Values: number of active neurons
    2: 62,
    3: 55,
    4: 50,
    5: 51,
    7: 53,
    8: 60,
    9: 43,
    10: 44,
    11: 53,
    12: 29,
    13: 17,
    14: 24,
    15: 59,
    16: 24,
    }


    file_of_model_to_load = model
    roi, layer = file_of_model_to_load.split('/')[-1].split('_')[2:4]
    roi = int(roi)
    roi_name = str(roi)
    if layer.isdigit():
        layer = int(layer)
    n_neurons=neurons_per_roi[roi]


    GPU = torch.cuda.is_available()
    net = VggModel(pretrained_model=vgg_pretrained, conv_layer=layer, num_neurons=n_neurons, device = torch.device("cuda:"+str(gpu_to_use) if GPU else "cpu"))
    net.cuda()

    if verbose: print('\n1/5  Model loading...')
    net.load_state_dict(torch.load(file_of_model_to_load, map_location=torch.device("cuda:"+str(gpu_to_use) if GPU else "cpu")))
    if verbose: print(net(torch.rand([14, 3, 224, 224]).cuda()).shape)

    # Untrained network, apparently need to have the line defining the pretrained network right before it or it doesnt work
    vgg_pretrained = vgg19(pretrained = True)
    net_untrained = VggModel(pretrained_model=vgg_pretrained, conv_layer=layer, num_neurons=n_neurons, device = torch.device("cuda:"+str(gpu_to_use) if GPU else "cpu"))
    net_untrained.cuda()

    if verbose: print('      Model loaded and tested...\n')



    '''
    ###############   2. Loading paolo model

    print('\n2/5   Loading Paolo\'s model')
    from neural_model import Model
    from lucent.modelzoo import inceptionv1, util, inceptionv1_avgPool
    import pickle
    import numpy as np

    def load_trained_model(roi_name,layer=False):
        if not layer:
            data_filename = '/home/jose/Desktop/snapshots/grid_search_array'+ roi_name +'.pkl'
            f = open(data_filename,"rb")
            cc = pickle.load(f)
            val_corrs = cc['val_corrs']
            params = cc['params']
            val_corrs = np.array(val_corrs)
            layer = params[np.where(val_corrs==val_corrs.max())[0][0].astype('int')][0]
        data_filename      = '/home/jose/Desktop/Data/data_THINGS_array'+ roi_name +'_v1.pkl'
        #data_filename = '/media/stijn/2bb74e85-3681-4561-88b7-abd98482de61/paolo/Data/data_THINGS_array'+ roi_name +'_v1.pkl'
        f = open(data_filename,"rb")
        cc = pickle.load(f)
        val_data = cc['val_data']
        del cc
        
        n_neurons = val_data.shape[1]

        pretrained_model = inceptionv1(pretrained=True)
        roi_model = Model(pretrained_model,layer,n_neurons,device='cpu')
        snapshot_path = f'/home/jose/Desktop/snapshots/array'+ roi_name +'_neural_model.pt'
        roi_model.load_state_dict(torch.load(snapshot_path,map_location=torch.device('cpu')))
        return roi_model,n_neurons


    model_paolo, n_neurons = load_trained_model(roi_name)
    print('      Loaded\n')
    '''


    ############ Loading the activations in the val and training data

    if verbose: print('\n3/5   Loading activations...')
    import pickle
    import numpy as np
    import itertools
    import scipy
    import matplotlib.pyplot as plt

    data_filename  = '/home/jose/Desktop/Data/data_THINGS_array'+ roi_name +'_v1.pkl'

    f = open(data_filename,"rb")
    cc = pickle.load(f)

    train_data = cc['train_data']
    val_data = cc['val_data']
    img_data = cc['img_data']
    val_img_data = cc['val_img_data']
    n_neurons = train_data.shape[1]
    if verbose: print('      Loaded\n')




    ########### Getting validation activations

    if verbose: print('\n4/5   Getting validation activations...')
    val_data_net   = net(val_img_data.cuda())
    val_data_net   = val_data_net.cpu().detach().numpy()


    '''
    val_data_paolo = model_paolo(val_img_data)
    val_data_paolo = val_data_paolo.cpu().detach().numpy()
    val_data_paolo = val_data_paolo[:,:,0,0]
    '''

    val_untrained = net_untrained(val_img_data.cuda())
    val_untrained = val_untrained.cpu().detach().numpy()
    if verbose: print('      Done\n')


    if verbose: print('\n5/5   Plotting...')
    f, ax = plt.subplots(10,10, figsize = (30,10))
    idx = 2

    f.suptitle('ROI: ' + str(roi_name) + '  Regularization L1\n Black: ground truth ; Red: Me\'s ')
    for i in range(100):
        ax[i//10, i%10].plot(val_data[i, :], 'k', label = 'Data')
        ax[i//10, i%10].plot(val_data_net[i, :], 'r--', label = 'Network')
        #ax[i//10, i%10].plot(val_data_paolo[i, :], 'b:', label = 'Paolo')
        ax[i//10, i%10].set_title('Image ' + str(i+1))

    #plt.savefig('Roi_3_layer_10_val_activations.png', dpi = 150)
    if verbose: print('      Done\n')
    plt.show()


    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################

    correlations = []

    for i in range(val_data.shape[1]):
        correlations.append(np.corrcoef(val_data[:, i], val_data_net[:, i])[1,0])
    correlations = np.array(correlations)

    return correlations