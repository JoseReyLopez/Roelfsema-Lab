from torch.utils.data import Dataset
from torchvision.io import read_image


class Images_MUA(Dataset):
    def __init__(self, mua_number, train = 1):
        print('Creating the dataloader')
        import h5py
        if type(mua_number) == int:
            print('home/jose/Desktop/Data/THINGS_exportMUA_array'+ str(mua_number) +'.mat')
            mua = h5py.File('home/jose/Desktop/Data/THINGS_exportMUA_array'+ str(mua_number) +'.mat')
        if type(mua_number) == str:
            mua = h5py.File('home/jose/Desktop/Data/THINGS_exportMUA_array'+ mua_number +'.mat')

        if train: 
            print('Train dataloader')
            self.activations = mua['train_MUA'][:, :, 0]
            self.img_dir = 'home/jose/Desktop/Data/THINGS_imgs/train/THINGS_train/'
        else:
            print('Test dataloader')
            self.activations = mua['test_MUA'][:, :, 0]
            self.img_dir = 'home/jose/Desktop/Data/THINGS_imgs/val/test/'
        
        
        print('\n\n\n\n')

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        img_path = self.img_dir + '/' + (5-len(str(idx)))*'0' + str(idx) + '.bmp'
        print('------  ', img_path)
        #image = read_image(img_path)
        #activation = self.activations[idx, :]
        #return image, activation