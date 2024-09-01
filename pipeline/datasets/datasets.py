import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
from ..datasets.preprocessing import ToTensor
import torch
import numpy as np
import glob
import random
import os

# Custom batch agregator that take multisize images for dataset that return (im,label,name,item)
def multisize_collate_fn(batch):
    images, labels, names, items = zip(*batch)
    sorted_indices = sorted(range(len(images)), key=lambda i: images[i].shape[-1]+images[i].shape[-2])
    sorted_images = [images[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_names = [names[i] for i in sorted_indices]
    sorted_items = [items[i] for i in sorted_indices]

    stacked_images = []
    current_size = None
    current_group = []
    for im in sorted_images:
        sh = im.shape
        if sh != current_size:
            if current_size is not None:
                tensor_group = torch.stack(current_group, dim=0)
                stacked_images.append(tensor_group)
            current_size = sh
            current_group = [im]
        else:
            current_group.append(im)
    if current_group:
        tensor_group = torch.stack(current_group, dim=0)
        stacked_images.append(tensor_group)

    sorted_labels = default_collate(sorted_labels)
    sorted_names = default_collate(sorted_names)
    sorted_items = default_collate(sorted_items)

    return [stacked_images,sorted_labels,sorted_names,sorted_items]



# load mstar image with label
# format : Dir -> target type -> data in npy
class mstar_data(data.Dataset):
    """ Store the eval images (1xHxWxC) and get normalized version"""
    def __init__(self, paths, process_func):
        
        self.Transform = process_func
        self.path_ = paths

    def __len__(self):
        return len(self.path_)

    def __getitem__(self, item):
        path = self.path_[item]
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        label = os.path.basename(os.path.dirname(path))

        im = np.abs(np.load(path))
        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
        assert (len(im.shape) == 3)
        x = self.Transform(im)
        
        return x,label,name,item

    def get_all_labels(self):
        labels = []
        for p in self.path_:
            label = os.path.basename(os.path.dirname(p))
            labels.append(label)
        return labels
    
# load mstar image with label
# format : Dir -> target type -> data in npy
class data_classification(data.Dataset):
    """ Store the eval images (1xHxWxC) and get normalized version"""
    def __init__(self, paths, process_func,classes):
        
        self.Transform = process_func
        self.path_ = paths
        self.classes = classes
        self.label_encoder = {class_name: torch.eye(len(self.classes))[i] for i, class_name in enumerate(self.classes)}
        self.prepare_labels()

    def prepare_labels(self):
        self.labels = []
        for path in self.path_:
            label = self.label_encoder[os.path.basename(os.path.dirname(path))]
            self.labels.append(label)

    def __len__(self):
        return len(self.path_)

    def __getitem__(self, item):
        label = self.labels[item]
        path = self.path_[item]
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        im = np.abs(np.load(path))
        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
        assert (len(im.shape) == 3)
        x = self.Transform(im)
        return x,label,name,item

    def get_all_labels(self):
        return self.labels, self.label_encoder
    

class data_classification_sample(data.Dataset):
    """ Store the eval images (1xHxWxC) and get normalized version"""
    def __init__(self, paths,process_func_real,process_func_synth,classes):
        
        self.Transform_real = process_func_real
        self.Transform_synth = process_func_synth
        self.path_ = paths
        self.classes = classes
        self.label_encoder = {class_name: torch.eye(len(self.classes))[i] for i, class_name in enumerate(self.classes)}
        self.prepare_labels()

    def prepare_labels(self):
        self.labels = []
        for path in self.path_:
            label = self.label_encoder[os.path.basename(os.path.dirname(path))]
            self.labels.append(label)

    def __len__(self):
        return len(self.path_)

    def __getitem__(self, item):
        label = self.labels[item]
        path = self.path_[item]
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        im = np.abs(np.load(path))
        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
        assert (len(im.shape) == 3)
        if '_synth_' in name:
            x = self.Transform_synth(im)
        elif '_real_' in name:
            x = self.Transform_real(im)
        return x,label,name,item

    def get_all_labels(self):
        return self.labels, self.label_encoder

# load patches and their labels (n_patches x H x W x n_ch or n_labels)
class data_segmentation(data.Dataset):
    """ Store the eval images (1xHxWxC) and get normalized version"""
    def __init__(self, patches, labels, process_func, n_labels):
        self.Transform = process_func
        self.Transform_labels = ToTensor()
        self.patches = patches
        self.labels = labels
        self.n_labels = n_labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, item):
        label = self.Transform_labels(self.labels[item])
        im = self.patches[item]
        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
        assert (len(im.shape) == 3)
        im,label = self.Transform(im,label)
        
        return im,label,item

class test_data(data.Dataset):
    """ Store the eval images (1xHxWxC) and get normalized version"""
    def __init__(self, datasets, process_func):
    
        self.Transform = process_func
        self.datasets_ = datasets
        len_ = 0
        for key in datasets:
            len_ += datasets[key]['len']
        self.len_ = len_

    def __len__(self):
        return self.len_

    def __getitem__(self, item):
        keys = [key for key in self.datasets_] 
        count = item
        pile = -1 # the pile where the image is stored
        stop = False
        while not stop:
            pile += 1
            pile_len = self.datasets_[keys[pile]]['len']
            stop = count<pile_len
            if not stop:
                count -= pile_len
        param_norm = self.datasets_[keys[pile]]['normalization'] # get normalization parameters for the dataset
        path = self.datasets_[keys[pile]]['im_paths'][count]
        
        label = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        im = np.abs(np.load(path).astype(np.float32))
        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
        assert (len(im.shape) == 3)

        x = self.Transform(im,param_norm)
        return (x,name,label)

    
class train(data.Dataset):
    """ Get a patch """
    def __init__(self, datasets, process_func=None,process_ref=None):
        self.datasets_ = datasets
        self.Transform = process_func

        len = 0
        for key in datasets:
            len += datasets[key]['len']
        self.len_ = len

    def __len__(self):
        return self.len_

    def __getitem__(self, item):
        keys = [key for key in self.datasets_] # name of dirs
        count = item # patch number in the pile
        pile = -1 # the pile where the image is stored
        stop = False
        while not stop:
            pile += 1
            pile_len = self.datasets_[keys[pile]]['len']
            stop = count<pile_len
            if not stop:
                count -= pile_len
        # print('item : {}, pile {}, count {}'.format(item,keys[pile],count))
        param_norm = self.datasets_[keys[pile]]['normalization'] # get normalization parameters for the dataset
        param_subsample = self.datasets_[keys[pile]]['subsample'] # boolean to know if subsample is performed
        im_slc,im_denoised = self.datasets_[keys[pile]]['slc'][count], self.datasets_[keys[pile]]['denoised'][count]
        x = self.Transform(im_slc,im_denoised,param_norm,param_subsample)
        return (x,param_norm)

class patch_pos_dataset(data.Dataset):
    def __init__(self,patches,process_fun,norm_param):
        self.patches = patches
        self.process_fun = process_fun
        self.norm_param = norm_param
    def __len__(self):
        return len(self.patches)
    def __getitem__(self,item):
        pos = self.patches[item]['pos']

        patch = self.patches[item]['patch']
        patch = self.process_fun(patch[:,:,np.newaxis],self.norm_param)
        return patch,pos


    
    
    
    
    
    
    
    
    