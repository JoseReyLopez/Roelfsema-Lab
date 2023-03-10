def plot_training_summary(file_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    roi, layer = file_name.split('.')[0].split('_')[-2:]
    tr_data = np.load(file_name)

    batchs, cols = tr_data.shape
    batchs = batchs - 1
    batchs_per_epoch = int(tr_data[:, 3][-1])
    epochs = int(batchs/batchs_per_epoch)

    locations = [batchs_per_epoch*i for i in range(epochs+1)]


    #print(locations)
    #print(batchs, batchs_per_epoch, epochs)


    f, ax = plt.subplot_mosaic('''
                       AAB
                       AA.
                       CDE
                       ''', figsize = (20,20))

    f.patch.set_facecolor('white')


    #f.suptitle(f'Training summary\n Epochs: {str(int(epochs))}     Batchs per epoch: {batchs_per_epoch}\n min T: {np.min(tr_data[1:, 4]):.3f}  -   min V: {np.min(tr_data[1:, 5]):.3f}', fontsize = 32)

    plt.text(x=0.5, y=0.94, s=f"Traning summary   -   Roi: {roi}   Layer: {layer}", fontsize=18, ha="center", transform=f.transFigure)
    plt.text(x=0.5, y=0.92, s=f" Epochs: {str(int(epochs))}     Batchs per epoch: {batchs_per_epoch}", fontsize=12, ha="center", transform=f.transFigure)
    plt.text(x=0.5, y=0.90, s=f"min T: {np.min(tr_data[1:, 4]):.3f}  -   min V: {np.min(tr_data[1:, 5]):.3f}", fontsize=12, ha="center", transform=f.transFigure)


    for i in locations:
        ax['A'].axvline(i, color = 'k', lw = .5)
    ax['A'].plot(tr_data[1:, 4], 'k', label = 'Training loss')
    ax['A'].plot(tr_data[1:, 5], 'r', lw = 5, label = 'Validation loss')
    ax['A'].set_title('Loss')
    ax['A'].legend()
    ax['A'].set_xticks(locations, [str(i) for i in range(epochs+1)])
    ax['A'].set_xlim(0, locations[-1])

    for i in locations:
        ax['B'].axvline(i, color = 'k', lw = .5)
    ax['B'].semilogy(tr_data[1:, 6], 'k')
    ax['B'].set_title('Learning rate')
    ax['B'].set_xticks(locations, [str(i) for i in range(epochs+1)])
    ax['B'].set_xlim(0, locations[-1])


    for i in locations:
        ax['C'].axvline(i, color = 'k', lw = .5)
    ax['C'].plot(tr_data[1:, 7], 'k')
    ax['C'].set_title('L1 loss')
    ax['C'].set_xticks(locations, [str(i) for i in range(epochs+1)])
    ax['C'].set_xlim(0, locations[-1])


    for i in locations:
        ax['D'].axvline(i, color = 'k', lw = .5)
    ax['D'].plot(tr_data[1:, 8], 'k')
    ax['D'].set_title('Smooth loss')
    ax['D'].set_xticks(locations, [str(i) for i in range(epochs+1)])
    ax['D'].set_xlim(0, locations[-1])


    for i in locations:
        ax['E'].axvline(i, color = 'k', lw = .5)
    ax['E'].plot(tr_data[1:, 9], 'k')
    ax['E'].set_title('Sparsity loss')
    ax['E'].set_xticks(locations, [str(i) for i in range(epochs+1)])
    ax['E'].set_xlim(0, locations[-1])

    if os.name == 'posix':
        #plt.savefig('/home/jose/Desktop/models/Roi_'+str(roi_name)+'_layer_'+str(layer)+'__'+str(int(time.time()))+'__epoch_'+str(epoch)+'.png')
        #plt.savefig(str(roi_name)+'_layer_'+str(layer)+'__'+str(int(time.time()))+'__epoch_'+str(epoch)+'.png')
        plt.savefig(f'training_summary_{roi}_layer_{layer}.png', transparent = False)
    if os.name == 'nt':
        #plt.savefig('C:/Users/lopez/Desktop/lucent-things/Roi_'+str(roi_name)+'_layer_'+str(layer)+'__'+str(int(time.time()))+'__epoch_'+str(epoch)+'.png')
        #plt.savefig(str(roi_name)+'_layer_'+str(layer)+'__'+str(int(time.time()))+'__epoch_'+str(epoch)+'.png')
        plt.savefig(f'training_summary_{roi}_layer_{layer}.png', transparent = False)

        