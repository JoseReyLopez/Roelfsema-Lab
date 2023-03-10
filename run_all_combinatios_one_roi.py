import os

#pairs = [[9, 7], [9, 10], [9, 12], [9, 23], [9, 25], [9, 28], [7,2], [7,5], [7,7], [7, 19], [7,21],[7, 23]]
#pairs = [[3, 14],[3,16], [4,12],[4,14],[8,21], [9, 5], [8,23], [8, 25], [9, 30],[12, 30],[13, 25], [14, 28], [14, 32], [14,34],[14, 25]] 
#pairs = [[16,34],[5,7],[5,10],[5,12],[5,14],[4,16],[5,7],[5,16],[11,16],[11,28],[15,25],[16,25],[3,19],[4,16],[4,19]] 
#pairs = [[10,14], [10,25],[12,34],[2,21],[4,21],[5, 19], [5,21],[7,25], [10,12],[11,14],[11,12],[10,23],[10,28]]
#pairs = [[12,19],[12,16],[12,14],[13,23],[13,21],[13,19], [14,23],[14,21],[14,19],[15,23],[15,21],[15,19],[16,23],[16,21],[16,19],[3,23]]
pairs = [[3,25],[4,25],[5,25]]

for n, (roi, layer) in enumerate(pairs):
    print(n, '/', len(pairs))
    print('python3 TRAIN_cadena_model_vgg.py -r '+str(roi)+' -l '+str(layer)+' -e '+str(10)+' -spa 1e-3 -smo 1e-3 -gsp 1e-2')
    try:
         os.system('python3 TRAIN_cadena_model_vgg.py -r '+str(roi)+' -l '+str(layer)+' -e '+str(10)+' -spa 1e-3 -smo  1e-3 -gsp 1e-2')
    except:
        continue