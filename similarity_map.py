import os

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pipeline.models.ViT as vits
from pipeline.utils import load_model, draw_progress_bar, plot_im
from pipeline.datasets.datasets import test_data
from pipeline.datasets.load import eval_data
from pipeline.datasets.preprocessing import *

def save_basic_with_colorbar(im, fold):
    # Create a figure and axis to display the image
    fig, ax = plt.subplots()
    # Display the image
    cax = ax.imshow(im, cmap='gray', vmin=0, vmax=1)
    # Add a colorbar next to the image
    fig.colorbar(cax, ax=ax, orientation='vertical')
    # Remove axis labels
    ax.axis('off')
    # Save the figure to a file
    plt.savefig(fold, bbox_inches='tight', pad_inches=0)
    # Close the plot to free memory
    plt.close()

def save_basic(im,fold):
    assert len(im.shape)==2
    im = im*255
    im = Image.fromarray(im.astype(np.uint8))
    im.save(fold)
        

def enlarge_pixels(image, p):
    enlarged_image = np.repeat(image, p, axis=0)
    enlarged_image = np.repeat(enlarged_image, p, axis=1)
    return enlarged_image

def display(features,feat_ref,params):
    # Number of vector to reduce
    nb_token = 0
    for im in features:
        shape = im['features'].shape
        nb_token += shape[0]*shape[1]
    token_sz = im['features'].shape[2]
    print('Data : {} vectors of size {}\n'.format(nb_token,token_sz))
    # Concatenating vector 
    x = np.empty((nb_token,token_sz))
    start_id = 0
    for im in features:
        shape = im['features'].shape
        end_id = start_id+shape[0]*shape[1]
        x[start_id:end_id] = im['features'].reshape((shape[0]*shape[1],token_sz))
        start_id = end_id   
    # Data similarity
    x_reduced = x@feat_ref.T
    
    # Display the similarity    
    start_id = 0
    for im in features:
        shape = im['features'].shape
        end_id = start_id+shape[0]*shape[1]
        features_im = x_reduced[start_id:end_id]
        features_im = features_im.reshape((shape[0],shape[1]))
        start_id = end_id
        if params['stride'] != 'full':
            features_im = enlarge_pixels(features_im,im['stride'])
        np.save('{}/{}_p{}_s{}.npy'.format(params['save_fold'],im['name'],im['patch_size'],im['stride']),features_im)
        save_basic(features_im,'{}/{}_p{}_s{}.png'.format(params['save_fold'],im['name'],im['patch_size'],im['stride']))
        thresh = 0.8
        features_im[features_im<thresh] = 0
        features_im[features_im>=thresh] = 1
        save_basic(features_im,'{}/{}_p{}_s{}_t{}.png'.format(params['save_fold'],im['name'],im['patch_size'],im['stride'],thresh))
        

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
        output = torch.squeeze(output) # H,W,D
        output = nn.functional.normalize(output, dim=2, p=2)
        output = output.data.numpy().astype(np.float32)
        # Data storing
        outputs.append({'name' : name,
                        'stride' : stride,
                        'patch_size' : patch_sz,
                        'features' : output })
        count += 1

    ref = np.load(params['ref_patch'])
    # ref = ref[16:-16,16:-16]
    ref = transform(ref[:,:,np.newaxis],params['ref_norm'])
    ref = ref[None,:,:,:].to(device)
    feat_ref = model(ref)
    feat_ref = torch.squeeze(feat_ref)
    feat_ref = nn.functional.normalize(feat_ref, dim=0, p=2)
    feat_ref = feat_ref.cpu().data.numpy().astype(np.float32)
    return outputs,feat_ref

if __name__ == '__main__':

    params = {}
    # Save and load folders
    params['save_fold'] = './data/similarity/results/'
    params['load_fold'] = './data/hrsid/eval'
    params['ref_patch'] = './data/hrsid/ref/boat.npy'
    params['ref_norm'] = [-2.,10.] 
    # Dataset parameters
    params['device'] = 'cuda'
    params['patch_sz'] = 64
    params['stride'] = 4 # auto to set stride = patch_size, or a positive int to set a custom stride
    # Network parameters
    params['weights_path'] = './pipeline/out/encoder_1ch'
    params['arch'] = 'vit_tiny'
    params['arch_patch_sz'] = 8
    params['channels'] = 1
    params['drop_rate'] = 0.
    params['batch_sz'] = 512
    # Display parameters 
    assert params['channels'] in [1,4]   
    os.makedirs(params['save_fold'],exist_ok=True)

    features,feat_ref = extract_feature_pipeline(params)
    display(features,feat_ref,params)



    