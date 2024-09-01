import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torchvision import transforms 
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import time

import pipeline.models.ViT as vits
from pipeline.utils import load_model, draw_progress_bar, save_plot
from pipeline.datasets.datasets import test_data
from pipeline.datasets.load import eval_data
from pipeline.datasets.preprocessing import *

def convert_str_list_to_numbers(str_list):
    str_to_number_mapping = {}
    result_numbers = []
    for s in str_list:
        if s not in str_to_number_mapping:
            # Assign a new number if the string is encountered for the first time
            str_to_number_mapping[s] = len(str_to_number_mapping) + 1
        result_numbers.append(str_to_number_mapping[s])
    return result_numbers

def display(df,params):
    final_dim = params['final_dim'] # 2 or 3
    print('Displaying lattent space features, projected with \'{}\' on {} dimensions ...'.format(params['display_type'],final_dim))
    # Data reduction
    if params['display_type'] == 'pca':
        pca_dim = final_dim
        pca = PCA(n_components=pca_dim)
        x_reduced = pca.fit_transform(df.filter(like='feature_',axis=1).values)
        print('Cumulative explained variation for {} principal components: {}'.format(pca_dim,np.sum(pca.explained_variance_ratio_)))
    elif params['display_type'] == 'tsne':
        tsne_dim = final_dim
        time_start = time.time()
        tsne = TSNE(n_components=tsne_dim, verbose=0, perplexity=40, n_iter=500)
        x_reduced = tsne.fit_transform(df.filter(like='feature_',axis=1).values)
        print('t-SNE done with {} components. Time elapsed: {} seconds'.format(tsne_dim,time.time()-time_start))
    elif params['display_type'] == 'pca_tsne':
        pca_dim = 30
        tsne_dim = final_dim
        # PCA
        pca = PCA(n_components=pca_dim)
        pca_result = pca.fit_transform(df.filter(like='feature_',axis=1).values)
        print('Cumulative explained variation for {} principal components: {}'.format(pca_dim,np.sum(pca.explained_variance_ratio_)))
        # t-SNE
        time_start = time.time()
        tsne = TSNE(n_components=tsne_dim, verbose=0, perplexity=40, n_iter=500)
        x_reduced = tsne.fit_transform(pca_result)
        print('t-SNE done with {} components. Time elapsed: {} seconds'.format(tsne_dim,time.time()-time_start))
    
    for i in range(final_dim):
        df['axis_{}'.format(i)] = x_reduced[:,i]
    # Display the components
    nb_label = len(pd.unique(df['label']))
    if final_dim == 2:
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="axis_0", y="axis_1",
            hue="name",
            palette=sns.color_palette("hls", nb_label),
            data=df,
            legend="full",
            alpha=0.7)
    elif final_dim == 3:
        # Visualisation 3D
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(
            xs=df["axis_0"], 
            ys=df["axis_1"], 
            zs=df["axis_2"], 
            c=df["label"])
        ax.set_xlabel('axis_0')
        ax.set_ylabel('axis_1')
        ax.set_zlabel('axis_2')
    
    plt.axis('off')
    plt.savefig(params['save_path'],bbox_inches = 'tight',pad_inches = 0)
    plt.clf()

@torch.no_grad()
def extract_feature_pipeline(params):
    # ============ preparing data ... ============
    device = params['device'] 
    patch_drop = params['drop_rate']
    data_ = eval_data(params['load_fold'])
    transform = TransformEvalSar()
    data = test_data(data_,transform)
    feature_data = DataLoader(data,batch_size=1,shuffle=False)
    vector_number = len(feature_data)
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
    embed_dim = model.embed_dim
    # ============ extract features ... ============
    print("Extracting features - total number of vectors : {}".format(vector_number))
    features = np.array([]).reshape(0,embed_dim)
    labels = []
    count = 0
    for im,_,label in feature_data :
        count += 1 
        draw_progress_bar(count,vector_number,'Feature exctraction')
        im = im.to(device)
        (un,c,h,w) = im.shape
        feat = torch.squeeze(model(im))
        features = np.vstack([features,feat.cpu().data.numpy().astype(np.float32)])
        labels.append(label)
    # features = features/np.linalg.norm(features, ord=2, axis=1, keepdims=True)

    features_col = ['feature_'+str(i) for i in range(features.shape[1])]
    labels_n = convert_str_list_to_numbers(labels)
    df = pd.DataFrame(features,columns=features_col)
    df['label'] = labels_n
    df['name'] = labels
    return df
        
        
if __name__ == '__main__':

    params = {}
    # Save and load folders
    params['save_path'] = './data/point_features/mstar'
    params['load_fold'] = './data/mstar'
    # Dataset parameters
    params['device'] = 'cuda'
    # Network parameters
    params['weights_path'] = './pipeline/out/encoder_1ch'
    params['arch'] = 'vit_tiny'
    params['arch_patch_sz'] = 8
    params['channels'] = 1
    params['drop_rate'] = 0.
    # Display parameters 
    params['final_dim'] = 2  # 2 or 3
    params['display_type'] = 'tsne'
        
    assert params['channels'] in [1,4]
    assert params['final_dim'] in [2,3]
    os.makedirs(os.path.dirname(params['save_path']), exist_ok=True)

    df = extract_feature_pipeline(params)
    display(df,params)