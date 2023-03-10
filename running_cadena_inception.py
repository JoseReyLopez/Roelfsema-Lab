import os
import itertools

os.chdir('/home/jose/Desktop/lucent-things')


vgg_layer_options = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]
# len = 16

inception_layer_options =  ['conv2d0', 'conv2d1', 'conv2d2', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b']
# len = 10

rois = list([2,3,4,5,7,8,9,10,11,12,13,14,15,16])



# Bashivan Vgg

for roi, inception_layer in itertools.product(rois, inception_layer_options):
    print(roi, inception_layer)

    try: 
        os.system(f'python TRAIN_cadena_model_inception.py -r {roi} -l {inception_layer}')
    except:
        pass