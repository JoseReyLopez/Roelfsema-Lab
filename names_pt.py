import os
from fnmatch import fnmatch
import glob
import itertools
import numpy as np

def select_folder(num1, num2):
    results = []
    for root, dirs, files in os.walk('Results'):
        for dir in dirs:
            if fnmatch(dir, f'ROI_{num1}_Layer_{num2}*'):
                subdir_path = os.path.join(root, dir)
                for subfiles in glob.glob(subdir_path+'/*'):
                    if subfiles.endswith('.pt'):
                        results.append(subfiles.rsplit('/',1)[-1])
    return results


hyperparam = sorted(select_folder(3,10))

print(hyperparam)

hyperparam_losses = np.zeros((len(hyperparam),4))

for n, config in enumerate(hyperparam):
    config = config.rsplit('.pt', 1)[0]
    print(config.split('_'))
    split_data = config.split('_')
    loss_value_index = split_data.index('loss')+1
    sparsity_loss_index = split_data.index('sparsity')+1
    smoothness_loss_index = split_data.index('smoothness')+1
    group_sparsity_loss_index = split_data.index('groupsparsity')+1

    hyperparam_losses[n,:] =split_data[loss_value_index], split_data[sparsity_loss_index], split_data[smoothness_loss_index], split_data[group_sparsity_loss_index]

    #print(split_data[loss_value_index], split_data[sparsity_loss_index], split_data[smoothness_loss_index], split_data[group_sparsity_loss_index])

print(hyperparam_losses)


sparsity = [10e-2, 10e-3, 10e-4]
smoothness = [10e-1, 10e-2, 10e-3]
groups_sparsity = [10e-3, 10e-4, 10e-5]

hyperparam_lists = [sparsity,smoothness, groups_sparsity]

for i in itertools.product(*hyperparam_lists):
    print(i)