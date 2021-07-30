# load, split and scale the maps dataset ready for training
from torch.utils.data import Dataset, DataLoader
from os import listdir
import cv2
from matplotlib import pyplot
import numpy as np


# load all images in a directory into memory


# dataset path
#path = '/content/drive/My Drive/data/pix2pix/maps/train/'
# load dataset
# print('Loaded: ', src_images.shape, tar_images.shape)
# # save as compressed numpy array
# filename = 'maps_256.npz'
# savez_compressed(filename, src_images, tar_images)
# print('Saved dataset: ', filename)
#size=(512, 256)

class MapDataset(Dataset):
    """MapDataset dataset."""

    def __init__(self, root_dir, resize=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.resize = resize
        
        self.files_list = listdir(self.root_dir)
        self.src_list, self.tar_list = __loadimages__()


    def __loadimages__(self): 
        srclist, tarlist = list(), list()       
        # enumerate filenames in directory
        for filename in files_list:
            # load and resize the image
            pixels = cv2.imread(self.root_dir + filename)
            pixels = cv2.resize(pixels, self.resize)
            
            # split into satellite and map
            sat_img, map_img = pixels[:, :256], pixels[:, 256:]
            self.srclist.append(sat_img)
            self.tarlist.append(map_img)

        return srclist, tarlist

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):

        X = transform(self.src_list[idx])        
        Y = transform(self.tar_list[idx])
        
        return X, Y



def dloader(datapath, resize, transform, shuffle):

    dset = MapDataset(datapath, resize, transform)
    loader = DataLoader(dset, batch_size=1, shuffle=None)

    return loader


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# visualize dataset samples
def view_sample():
    n_samples = 3
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(src_images[i].astype('uint8'))
    # plot target image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(tar_images[i].astype('uint8'))
    pyplot.show()