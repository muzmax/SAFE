import glob
import random
import os
import numpy as np
import gc
import shutil
from scipy import signal
import torch
from PIL import Image
from ..utils import normalize01, draw_progress_bar, disp_sar, plot_im
from ..datasets.preprocessing import normalization, numpy_resize
from ..logger import LOGGER
from torchvision import transforms

"""
Convert each path of a slc image to its corresponding denoised version
"""
def slc2denoised_path(files:list):
    paths_denoised = []
    fold = os.path.split(files[0])[0] 
    fold = os.path.split(fold) 
    fold = os.path.join(fold[0],fold[1].replace('_slc','_denoised')) # change the name of the last dir
    for file in files:
        file_denoised = os.path.join(fold,'denoised_{}'.format(os.path.basename(file))) # change the name of the npy file
        paths_denoised.append(file_denoised)
    assert len(paths_denoised) == len(files)
    return paths_denoised

"""
input : directory of multiple sub directory
output : dictionary of datasets
"""
def train_data(dir, pat_size=256,stride=700,bat_size=32, nb_ch=1):
    assert nb_ch in [1,4]
    dataset = {}
    total_patches = 0
    dirpaths = glob.glob(dir+'/*_slc/') 
    
    if nb_ch == 4:
        dirpaths = list(filter(lambda f : 'sentinel_' not in f, dirpaths))
    dirpaths = list(filter(lambda f : 'sentinel_' not in f, dirpaths))  
    dirpaths = list(filter(lambda f : 'uav_' not in f, dirpaths))
    # dirpaths = list(filter(lambda f : 'Nimes_X_' not in f, dirpaths))
    dirpaths = list(filter(lambda f : 'dessous_X_' not in f, dirpaths))
    dirpaths = list(filter(lambda f : 'dessous_L_' not in f, dirpaths))
    LOGGER.info('Preparing data from {} folder(s)...'.format(len(dirpaths)))

    for dir in dirpaths:
        dir = os.path.normpath(dir)
        dirname = os.path.basename(dir).replace('_slc','')
        print(dirname,end=' - ')
        """ Get the normalization parameters for the dataset """
        with open(os.path.join(dir,'info.txt'),'r') as fin:
            for line in fin:
                if 'norm' in line:
                    first,last = line.find('['), line.find(']')
                    param = line[first+1:last].replace(' ','')
                    param = np.array([float(value) for value in param.split(',')])
                if 'subsample' in line:
                    first = line.find(':')
                    do_sub = int(line[first+1:].replace(' ',''))
                    assert do_sub in[0,1]
                    do_sub = bool(do_sub)
            assert param.shape == (2,)
            assert param.dtype == float
            assert type(do_sub) == bool
                
        if nb_ch == 1:
            data_slc, data_denoised = generate_patches_1ch(dir,pat_size=pat_size,stride=stride,bat_size=bat_size)
        elif nb_ch == 4:
            data_slc, data_denoised = generate_patches_4ch(dir,pat_size=pat_size,stride=stride,bat_size=bat_size)
        nb_patches = len(data_slc)
        total_patches += nb_patches
        dataset[dirname] = {'len' : len(data_slc),
                            'slc' : data_slc,
                            'denoised' : data_denoised,
                            'normalization' : param,
                            'subsample' : do_sub}

    LOGGER.info('total nb of patches : {}. total nb of batches : {}.'.format(total_patches,int(total_patches/bat_size)))
    return dataset

"""
input : directory of multiple sub directory
output : dictionary of paths
"""
def eval_data(dir):
    dataset = {}
    dirpaths = glob.glob(dir+'/*/') 
    print('Preparing data from {} folder(s)...'.format(len(dirpaths)))
    total_im = 0
    for dir in dirpaths:
        dir = os.path.normpath(dir)
        dirname = os.path.basename(dir)
        print(dirname,end=' - ')
        """ Get the normalization parameters for the dataset """
        with open(os.path.join(dir,'info.txt'),'r') as fin:
            for line in fin:
                if 'norm' in line:
                    first,last = line.find('['), line.find(']')
                    param = line[first+1:last].replace(' ','')
                    param = np.array([float(value) for value in param.split(',')])
            assert param.shape == (2,)
            assert param.dtype == float

        filepaths = glob.glob(os.path.join(dir,'*.npy'))
        nb_im = len(filepaths)
        print('{} image(s)'.format(nb_im))
        total_im += nb_im
        dataset[dirname] = {'len' : nb_im,
                            'im_paths' : filepaths,
                            'normalization' : param }
    print('total nb of images : {}.'.format(total_im))
    return dataset
            
"""
input : directory name
output : numpy matrix [numPatches, pat_size, pat_size, ch] of patches
"""
def generate_patches_1ch(src_dir="./data/train", pat_size=256, step=0, stride=16, bat_size=4):
        
        # paths for slc and their respective denoised image
        filepaths = glob.glob(os.path.join(src_dir,'*.npy'))
        filepaths_denoised = slc2denoised_path(filepaths)
        # calculate the number of patches
        count = 0
        print('{} image(s)'.format(len(filepaths)),end=' - ')
        for i in range(len(filepaths)):
            img = np.load(filepaths[i])
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            for x in range(0+step, (im_h - pat_size), stride):
                for y in range(0+step, (im_w - pat_size), stride):
                    count += 1
        origin_patch_num = count 
        if origin_patch_num % bat_size != 0:
            numPatches = origin_patch_num + (bat_size - origin_patch_num % bat_size)
        else:
            numPatches = origin_patch_num
        numPatches=int(numPatches)
        print("nb of patches = {}".format(numPatches))

        """ 
        SLC data 
        """
        print('Preparing slc data.',end=' ')
        # data matrix 4-D
        data_slc = np.zeros((numPatches, pat_size, pat_size, 1), dtype=np.complex64)  
        # generate patches
        count = 0
        for i in range(len(filepaths)): #scan through images
            img = np.load(filepaths[i])
            img = np.array(img, dtype=np.complex64)
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            img = np.reshape(img, (np.size(img, 0), np.size(img, 1), 1))  # extend one dimension
            for x in range(0 + step, im_h - pat_size, stride):
                for y in range(0 + step, im_w - pat_size, stride):
                    data_slc[count, :, :, :] = img[x:x + pat_size, y:y + pat_size, :]
                    count += 1
        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            data_slc[-to_pad:, :, :, :] = data_slc[:to_pad, :, :, :]

        """ 
        Denoised data 
        """
        print('Preparing denoised data.')
        # data matrix 4-D
        data_denoised = np.zeros((numPatches, pat_size, pat_size, 1), dtype="float32") 
        
        # generate patches
        count = 0
        for i in range(len(filepaths_denoised)): #scan through images
            img = np.load(filepaths_denoised[i])
            img = np.abs(img)
            img = np.array(img, dtype="float32")
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            img = np.reshape(img, (np.size(img, 0), np.size(img, 1), 1))  # extend one dimension
            for x in range(0 + step, im_h - pat_size, stride):
                for y in range(0 + step, im_w - pat_size, stride):
                    data_denoised[count, :, :, :] = img[x:x + pat_size, y:y + pat_size, :]
                    count += 1
        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            data_denoised[-to_pad:, :, :, :] = data_denoised[:to_pad, :, :, :]
        
        return data_slc, data_denoised

"""
input : directory name
output : numpy matrix [numPatches, pat_size, pat_size, ch] of patches
"""
def generate_patches_4ch(src_dir="./data/train", pat_size=256, step=0, stride=16, bat_size=4):
        filepaths = glob.glob(os.path.join(src_dir,'*.npy'))
        # paths for slc and their respective denoised image
        filepaths_hv = list(filter(lambda f : 'hv' in os.path.basename(f).lower(),filepaths))
        filepaths_vh = list(filter(lambda f : 'vh' in os.path.basename(f).lower(),filepaths))
        filepaths_hh = list(filter(lambda f : 'hh' in os.path.basename(f).lower(),filepaths))
        filepaths_vv = list(filter(lambda f : 'vv' in os.path.basename(f).lower(),filepaths))
        filepaths_denoised_hv = slc2denoised_path(filepaths_hv)
        filepaths_denoised_vh = slc2denoised_path(filepaths_vh)
        filepaths_denoised_hh = slc2denoised_path(filepaths_hh)
        filepaths_denoised_vv = slc2denoised_path(filepaths_vv)
        assert len(filepaths_hv) == len(filepaths_vh) == len(filepaths_hh) == len(filepaths_vv) == len(filepaths_denoised_hv) == len(filepaths_denoised_vh) == len(filepaths_denoised_hh) == len(filepaths_denoised_vv)
        # calculate the number of patches
        count = 0
        print('{} image(s)'.format(len(filepaths_hv)),end=' - ')
        for i in range(len(filepaths_hv)):
            img = np.load(filepaths_hv[i])
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            for x in range(0+step, (im_h - pat_size), stride):
                for y in range(0+step, (im_w - pat_size), stride):
                    count += 1
        origin_patch_num = count 
        if origin_patch_num % bat_size != 0:
            numPatches = origin_patch_num + (bat_size - origin_patch_num % bat_size)
        else:
            numPatches = origin_patch_num
        numPatches=int(numPatches)
        print("nb of patches = {}".format(numPatches))

        """ 
        SLC data 
        """
        print('Preparing polsar data.',end=' ')
        # data matrix 4-D
        data_slc = np.zeros((numPatches, pat_size, pat_size, 4), dtype=np.complex64) 
        # generate patches
        count = 0
        for i in range(len(filepaths_hh)): # scan through images and concatenate channels
            im = np.load(filepaths_hh[i])
            im = np.array(im, dtype=np.complex64)
            im_h = np.size(im, 0)
            im_w = np.size(im, 1)
            im_pol = np.empty((im_h,im_w,4),dtype=np.complex64)
            im_pol[:,:,0] = im
            im = np.load(filepaths_hv[i])
            im = np.array(im, dtype=np.complex64)
            im_pol[:,:,1] = im
            im = np.load(filepaths_vh[i])
            im = np.array(im, dtype=np.complex64)
            im_pol[:,:,2] = im
            im = np.load(filepaths_vv[i])
            im = np.array(im, dtype=np.complex64)
            im_pol[:,:,3] = im
            for x in range(0 + step, im_h - pat_size, stride):
                for y in range(0 + step, im_w - pat_size, stride):
                    data_slc[count, :, :, :] = im_pol[x:x + pat_size, y:y + pat_size, :]
                    count += 1
        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            data_slc[-to_pad:, :, :, :] = data_slc[:to_pad, :, :, :]

        """ 
        Denoised data 
        """
        print('Preparing denoised data.')
        # data matrix 4-D
        data_denoised = np.zeros((numPatches, pat_size, pat_size, 4), dtype=np.float32) 
        # generate patches
        count = 0
        for i in range(len(filepaths_denoised_hh)): # scan through images and concatenate channels
            im = np.load(filepaths_denoised_hh[i])
            im = np.array(im, dtype=np.float32)
            im_h = np.size(im, 0)
            im_w = np.size(im, 1)
            im_pol = np.empty((im_h,im_w,4),dtype=np.float32)
            im_pol[:,:,0] = im
            im = np.load(filepaths_denoised_hv[i])
            im = np.array(im, dtype=np.float32)
            im_pol[:,:,1] = im
            im = np.load(filepaths_denoised_vh[i])
            im = np.array(im, dtype=np.float32)
            im_pol[:,:,2] = im
            im = np.load(filepaths_denoised_vv[i])
            im = np.array(im, dtype=np.float32)
            im_pol[:,:,3] = im
            for x in range(0 + step, im_h - pat_size, stride):
                for y in range(0 + step, im_w - pat_size, stride):
                    data_denoised[count, :, :, :] = im_pol[x:x + pat_size, y:y + pat_size, :]
                    count += 1
        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            data_denoised[-to_pad:, :, :, :] = data_denoised[:to_pad, :, :, :]
        
        return data_slc, data_denoised


def load_eval_data(dir="./data/eval",label = False):
    
    filepaths = glob.glob(dir + '/*.npy')
    
    if label:
        imagepaths = list(filter(lambda file : 'mask' not in file,filepaths))
        maskpaths = list(filter(lambda file : 'mask' in file,filepaths))
        nb = len(imagepaths)
        assert(nb == len(maskpaths))
        print("Number of eval data with label = %d" %nb)
        return imagepaths,maskpaths

    nb = len(filepaths)
    print("Number of eval data without label = %d" %nb)
        
    return filepaths

def get_paths(filepaths:str,n_label:int,list_paths:list):
    filepaths.sort()
    total_im = 0
    nb_im = len(filepaths)

    if n_label == 0:
        return list_paths,0
    
    elif n_label == -1 or n_label >=nb_im:
        total_im += nb_im
        list_paths = [*list_paths,*filepaths]

    else :
        total_im += n_label
        if n_label == 1:
            list_paths.append(filepaths[0])
        elif n_label == 2:
            list_paths.append(filepaths[0])
            list_paths.append(filepaths[-1])
        elif n_label >= 3:
            step = (nb_im - 1) / (n_label - 1)
            list_paths.extend([filepaths[int(i * step)] for i in range(n_label)])
    return list_paths,total_im
        

def load_paths(dir,ratio=0.8,batch_sz=1):

    train_paths = []
    eval_paths = []
    total_im = 0
    sub_dirs = glob.glob('{}/*'.format(dir))
    classes = [class_name for class_name in os.listdir(dir) if os.path.isdir(os.path.join(dir, class_name))]

    for sub_dir in sub_dirs:
        filepaths = glob.glob('{}/*.npy'.format(sub_dir))
        nb_im = len(filepaths)
        total_im += nb_im
        ind = list(range(nb_im))
        random.shuffle(ind)
        train_sub = [filepaths[i] for i in ind[:int(nb_im*ratio)]]
        eval_sub = [filepaths[i] for i in ind[int(nb_im*ratio):]]
        train_paths = [*train_paths,*train_sub]
        eval_paths = [*eval_paths,*eval_sub]
    print("Found {} dirs with a total of {} images split in test train with a ratio of {}".format(len(sub_dirs),total_im,ratio))
    print('{} steps for the training phase'.format(int((total_im*ratio)//batch_sz)))
    return train_paths,eval_paths,classes

def load_separated_paths(dir1,dir2,n_label=-1,batch_sz=1):
    train_paths = []
    eval_paths = []
    sub_dirs_1 = glob.glob('{}/*'.format(dir1)) # train
    sub_dirs_2 = glob.glob('{}/*'.format(dir2)) # eval
    classes_1 = [class_name for class_name in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, class_name))]
    classes_2 = [class_name for class_name in os.listdir(dir2) if os.path.isdir(os.path.join(dir2, class_name))]
    assert classes_1 == classes_2

    # get training images
    total_im = 0
    for sub_dir in sub_dirs_1:
        filepaths = glob.glob('{}/*.npy'.format(sub_dir))
        train_paths,nb_im = get_paths(filepaths,n_label,train_paths)
        total_im += nb_im
    print("{} - Found {} dirs with a total of {} training images and {} steps".format(dir1,len(sub_dirs_1),total_im,int(total_im//batch_sz)))

    # get eval images
    total_im = 0
    for sub_dir in sub_dirs_2:
        filepaths = glob.glob('{}/*.npy'.format(sub_dir))
        nb_im = len(filepaths)
        total_im += nb_im
        eval_paths = [*eval_paths,*filepaths]
    print("{} - Found {} dirs with a total of {} evaluation images ".format(dir2,len(sub_dirs_2),total_im))

    return train_paths,eval_paths,classes_1

def load_separated_paths_sample(dir,n_label_real=-1,n_label_synth=-1,batch_sz=1,take_eval_synth=False):
    train_paths = []
    eval_paths = []
    train_real = glob.glob(os.path.join(dir,'real','train','*'))
    train_synth = glob.glob(os.path.join(dir,'synth','train','*'))
    eval_real = glob.glob(os.path.join(dir,'real','test','*'))
    eval_synth = glob.glob(os.path.join(dir,'synth','test','*'))

    classes_1 = [class_name for class_name in os.listdir(os.path.join(dir,'real','train')) if os.path.isdir(os.path.join(dir,'real','train', class_name))]
    classes_2 = [class_name for class_name in os.listdir(os.path.join(dir,'synth','train')) if os.path.isdir(os.path.join(dir,'synth','train', class_name))]
    classes_3 = [class_name for class_name in os.listdir(os.path.join(dir,'real','test')) if os.path.isdir(os.path.join(dir,'real','test', class_name))]
    classes_4 = [class_name for class_name in os.listdir(os.path.join(dir,'synth','test')) if os.path.isdir(os.path.join(dir,'synth','test', class_name))]
    assert classes_1 == classes_2 == classes_3 == classes_4
    total_im = 0
    # Get real training images
    for sub_dir in train_real:
        filepaths = glob.glob('{}/*.npy'.format(sub_dir))
        train_paths,nb_im = get_paths(filepaths,n_label_real,train_paths)
        total_im += nb_im
    # Get synth training images
    for sub_dir in train_synth:
        filepaths = glob.glob('{}/*.npy'.format(sub_dir))
        train_paths,nb_im = get_paths(filepaths,n_label_synth,train_paths)
        total_im += nb_im
    print("Found a total of {} training images and {} steps".format(total_im,int(total_im//batch_sz)))

    total_im = 0
    # Get real eval images
    for sub_dir in eval_real:
        filepaths = glob.glob('{}/*.npy'.format(sub_dir))
        nb_im = len(filepaths)
        total_im += nb_im
        eval_paths = [*eval_paths,*filepaths]
    # Get synth eval images
    if take_eval_synth:
        for sub_dir in eval_synth:
            filepaths = glob.glob('{}/*.npy'.format(sub_dir))
            nb_im = len(filepaths)
            total_im += nb_im
            eval_paths = [*eval_paths,*filepaths]
    print("Found a total of {} evaluation images ".format(total_im))

    return train_paths,eval_paths,classes_1

def load_segmentation(dir,rgb_id_dic,n_labels,input_dim = 256,batch_size=1,resize=True):
    f = numpy_resize(input_dim) # resize function
    label_encoder = {i: np.eye(n_labels)[i] for i in range(n_labels)} # id to one hot encoding
    # Extracting and sorting paths
    paths_HH = glob.glob(os.path.join(dir,'*HH.tiff'))
    paths_HV = glob.glob(os.path.join(dir,'*HV.tiff'))
    paths_VH = glob.glob(os.path.join(dir,'*VH.tiff'))
    paths_VV = glob.glob(os.path.join(dir,'*VV.tiff'))
    path_label = glob.glob(os.path.join(dir,'*.png'))
    paths_HH.sort()
    paths_HV.sort()
    paths_VH.sort()
    paths_VV.sort()
    path_label.sort()
    assert len(paths_HH) == len(paths_HV) == len(paths_VH) == len(paths_VV) == len(path_label)
    
    hh = np.array(Image.open(paths_HH[0]))
    if resize:
        LOGGER.info('Reshaping images of shape {} into shape {}.'.format(hh.shape,(input_dim,input_dim)))
    else:
        assert hh.shape == (input_dim,input_dim)
    nb_im = len(paths_HH)
    LOGGER.info('Loading {} images in {}.'.format(nb_im,dir))
    LOGGER.info('Total nb of batches : {}.'.format(nb_im//batch_size))

    # Crop 4 pol image and labels in patches
    patches = []
    labels = []
    n = 0
    for p_hh,p_hv,p_vh,p_vv,p_label in zip(paths_HH,paths_HV,paths_VH,paths_VV,path_label):
        n += 1
       # draw_progress_bar(n,total=nb_im)
        hh = np.array(Image.open(p_hh))
        hv = np.array(Image.open(p_hv))
        vh = np.array(Image.open(p_vh))
        vv = np.array(Image.open(p_vv))
        label = np.array(Image.open(p_label))
        pol = np.empty((*hh.shape,4),dtype=np.float32)
        pol[:,:,0] = hh.astype(np.float32)
        pol[:,:,1] = hv.astype(np.float32)
        pol[:,:,2] = vh.astype(np.float32)
        pol[:,:,3] = vv.astype(np.float32)
        if resize:
            pol = f(pol)
            label = f(label)
        patches.append(pol)
        encoded_label = np.zeros((input_dim,input_dim,n_labels),dtype=np.float32)
        for k in range(input_dim):
            for l in range(input_dim):
                label_id = rgb_id_dic[tuple(label[k,l,:])]
                if label_id in label_encoder.keys():
                    encoded_label[k,l,:] = label_encoder[label_id]
                else:
                    encoded_label[k,l,:] = np.repeat(None,n_labels) 
                    
        labels.append(encoded_label)
        
    return patches,labels

def load_patch_position(im_path,patch_sz,stride,batch_size):
    im = np.load(im_path)
    h,w = im.shape
    numPatches = 0
    for _ in range(0, (h - patch_sz), stride):
        for _ in range(0, (w - patch_sz), stride):
            numPatches += 1
    LOGGER.info('Number of patches : {}. Number of batches : {}.'.format(numPatches,int(numPatches/batch_size)))
    patches = []
    for x in range(0, (h - patch_sz), stride):
        for y in range(0, (w - patch_sz), stride):
            patches.append({'patch' : im[x:x+patch_sz,y:y+patch_sz],
                            'pos' : (x,y)})
    return patches



