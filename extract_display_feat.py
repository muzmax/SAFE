import os

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import time

import pipeline.models.ViT as vits
from pipeline.utils import load_model, disp_sar, draw_progress_bar
from pipeline.datasets.datasets import test_data
from pipeline.datasets.load import eval_data
from pipeline.datasets.preprocessing import *


denorm = denormalization(-2.5,12.)
def disp_debug(input_data: torch.Tensor):
    im = torch.squeeze(input_data[0,:,:,:]).cpu().data.numpy()
    if len(im.shape) == 3 :# multi polarization image
        im = im.transpose(1,2,0)
    im_d = denorm(im)
    print('\nbatch shape : {}'.format(input_data.shape))
    print('batch - mean {} - std {}'.format(torch.mean(input_data),torch.std(input_data)))
    print('im - mean {} - std {}'.format(np.mean(im),np.std(im)))
    print('im denorm - mean {} - std {}'.format(np.mean(im_d),np.std(im_d)))
    disp_sar(im_d,tresh=None)

def save_basic(im,fold):
    im = np.abs(im)*255
    assert len(im.shape)==3 and im.shape[2]==3
    im = Image.fromarray(im.astype(np.uint8), 'RGB')
    im.save(fold)
        

def enlarge_pixels(image, p):
    enlarged_image = np.repeat(image, p, axis=0)
    enlarged_image = np.repeat(enlarged_image, p, axis=1)
    return enlarged_image

def display(features,params):
    # Number of vector to reduce
    nb_token = 0
    for im in features:
        shape = im['features'].shape
        nb_token += shape[0]*shape[1]
    token_sz = im['features'].shape[2]
    print('Display type : {}\nData : {} vectors of size {}\n'.format(params['display_type'],nb_token,token_sz))
    # Concatenating vector 
    x = np.empty((nb_token,token_sz))
    start_id = 0
    for im in features:
        shape = im['features'].shape
        end_id = start_id+shape[0]*shape[1]
        x[start_id:end_id] = im['features'].reshape((shape[0]*shape[1],token_sz))
        start_id = end_id
            
    # Data reduction
    if params['display_type'] == 'pca':
        pca_dim = 3
        pca = PCA(n_components=pca_dim)
        x_reduced = pca.fit_transform(x)
        print('Cumulative explained variation for {} principal components: {}'.format(pca_dim,np.sum(pca.explained_variance_ratio_)))
    elif params['display_type'] == 'tsne':
        tsne_dim = 3
        time_start = time.time()
        tsne = TSNE(n_components=tsne_dim, verbose=0, perplexity=40, n_iter=500)
        x_reduced = tsne.fit_transform(x)
        print('t-SNE done with {} components. Time elapsed: {} seconds'.format(tsne_dim,time.time()-time_start))
    elif params['display_type'] == 'umap':
        umap_dim = 3
        umap_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=umap_dim,  
            metric='euclidean'  # or 'cosine'
        )
        x_reduced = umap_model.fit_transform(x)
    elif params['display_type'] == 'pca_tsne':
        pca_dim = 50
        tsne_dim = 3
        # PCA
        pca = PCA(n_components=pca_dim)
        pca_result = pca.fit_transform(x)
        print('Cumulative explained variation for {} principal components: {}'.format(pca_dim,np.sum(pca.explained_variance_ratio_)))
        # t-SNE
        time_start = time.time()
        tsne = TSNE(n_components=tsne_dim, verbose=0, perplexity=40, n_iter=500)
        x_reduced = tsne.fit_transform(pca_result)
        print('t-SNE done with {} components. Time elapsed: {} seconds'.format(tsne_dim,time.time()-time_start))
    # Normalize the 3 components
    for ch in range(x_reduced.shape[1]):
            x_reduced[:,ch] = x_reduced[:,ch]/np.amax(x_reduced[:,ch])
    # Display the 3 components
    start_id = 0
    for im in features:
        shape = im['features'].shape
        end_id = start_id+shape[0]*shape[1]
        features_im = x_reduced[start_id:end_id]
        features_im = features_im.reshape((shape[0],shape[1],3))
        start_id = end_id
        if params['stride'] != 'full':
            features_im = enlarge_pixels(features_im,im['stride'])
        save_basic(features_im,'{}/{}_{}_p{}_s{}.png'.format(params['save_fold'],im['name'],params['display_type'],im['patch_size'],im['stride']))
        np.save('{}/{}_{}_p{}_s{}.npy'.format(params['save_fold'],im['name'],params['display_type'],im['patch_size'],im['stride']),features_im)
        

@torch.no_grad()
def extract_feature_pipeline(params):
    # ============ preparing data ... ============
    device = params['device'] 
    patch_sz = params['patch_sz']
    patch_drop = params['drop_rate']
    data_ = eval_data(params['load_fold'])
    transform = TransformEvalSar()
    data = test_data(data_,transform)
    feature_data = DataLoader(data,batch_size=1,shuffle=False)
    # ============ building network ... ============
    print("Loading model : {}\nweights path : {}\nchannel(s) : {} \ndrop rate : {} ".format(params['arch'],params['weights_path'],params['channels'],params['drop_rate']))
    if params['arch'] in vits.__dict__.keys():
            model = vits.__dict__[params['arch']](
                patch_size = params['arch_patch_sz'],
                in_chans = params['channels'])  
            model.set_patch_drop(patch_drop)
    model.to(device)
    load_model(model, params['weights_path'])
    model.eval()
    # ============ Choosing stride option ... ============
    if params['stride'] == 'auto':
        stride = patch_sz
    elif type(params['stride']) == int and params['stride'] >= 1:
        stride = params['stride']
    else :
        print('Stride parameter must be \'auto\' or a positive integer, not {}'.format(params['stride']))
    # ============ extract features ... ============
    nb_im = len(data)
    print("Extracting features - \n number of images : {}\npatch size : {}\ stride : {}".format(nb_im,params['patch_sz'],stride))
    outputs = []
    count = 1
    for im,name,_ in feature_data :
        draw_progress_bar(count,nb_im,'Feature exctraction - {}'.format(name))
        name = name[0]
        b,c,h,w = im.shape
        # Number of patches along each dimension
        num_patches_H = h // stride
        num_patches_W = w // stride
        # Total length required to fit all patches exactly
        total_length_H = patch_sz + (num_patches_H - 1) * stride
        total_length_W = patch_sz + (num_patches_W - 1) * stride
        # Padding needed: difference between required total length and original dimension
        pad_h = total_length_H - h
        pad_w = total_length_W - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        im = F.pad(im, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        # Slicing the image in patches
        im = im.unfold(2, patch_sz, stride).unfold(3, patch_sz, stride)
        im = im.contiguous().view(b, c, -1, patch_sz, patch_sz)
        im = im.permute(0, 2, 1, 3, 4).reshape(-1, c, patch_sz, patch_sz)
        # Extracing features
        patch_outputs = []
        for i in range(0, im.size(0), params['batch_sz']):
            output = model(im[i:i + params['batch_sz']].to(device))
            patch_outputs.append(output.cpu())
        output = torch.cat(patch_outputs, dim=0)
        # Resizing to keep the spatial dimension
        output_h = num_patches_H
        output_w = num_patches_W
        output = output.reshape(b, output_h, output_w, output.shape[-1])
        output = torch.squeeze(output.permute(0, 1, 2, 3)) # H,W,D
        output = nn.functional.normalize(output, dim=2, p=2)
        output = output.data.numpy().astype(np.float32)
        # Data storing
        outputs.append({'name' : name,
                        'stride' : stride,
                        'patch_size' : patch_sz,
                        'features' : output })
        count += 1
    return outputs

if __name__ == '__main__':

    params = {}
    # Save/load folders
    params['save_fold'] = '/data/display_features/results'
    params['load_fold'] = '/data/display_features/eval'
    # Dataset parameters
    params['device'] = 'cuda'
    params['patch_sz'] = 64
    params['stride'] = 4 # auto to set stride = patch_size, or a positive int to set a custom stride
    # Network parameters
    params['weights_path'] = 'pipeline/out/encoder_1ch'
    params['arch'] = 'vit_tiny'
    params['arch_patch_sz'] = 8
    params['channels'] = 1
    params['drop_rate'] = 0.
    params['batch_sz'] = 256
    # Display parameters 
    params['display_type'] = 'umap'

    assert params['channels'] in [1,4]   
    os.makedirs(params['save_fold'],exist_ok=True)

    features = extract_feature_pipeline(params)
    display(features,params)



    