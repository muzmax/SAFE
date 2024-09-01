import os

import pipeline.models.ViT as vits
from pipeline.utils import load_model
from pipeline.predictor.mstar_knn import mstar_knn
from pipeline.logger import setup_logger,LOGGER

import torch.nn as nn
from torchvision import transforms 

import numpy as np

class no_deep(nn.Module):
    def __init__(self,resize_dim):
        super().__init__()
        self.rs_dim = resize_dim
        self.rs = transforms.CenterCrop((resize_dim,resize_dim))
    def forward(self, x):
        b,c,_,_ = x.shape
        out = self.rs(x)
        return out.contiguous().view(1,self.rs_dim**2)

def eval_knn(params):
    # ============ building logger ... ============
    os.makedirs(os.path.dirname(params['logger_path']),exist_ok=True)
    setup_logger(out_file=params['logger_path'],setup_msg=False)
    # ============ building network and dataset pipeline ... ============

    if not(params['no_network']):
        if params['arch'] in vits.__dict__.keys():
                model = vits.__dict__[params['arch']](patch_size=params['patch_size'])
                model.set_patch_drop(params['patch_drop'])
                model.to(params['device'])
                load_model(model, params['model_path'])
                model.eval()
        else :
            print("Unknow architecture: {}".format(params['arch']))
    else :
        model = no_deep(params['resize_dim'])

    eval = mstar_knn(params['normalization'],
                           params['eval_path'],
                           params['labeled_path'],
                           n_label = params['n_label'],
                           reduction=params['reduction'],
                           red_dim=params['red_dim'],
                           red_type=params['red_type'],
                           device=params['device'],
                           no_network = params['no_network'])
    # ============ evaluation ... ============
    results = eval(model, nb_knn=params['nb_knn'], temperature=params['temperature'])
    LOGGER.info('Results : {}'.format(results))


if __name__ == '__main__':
    params = {}
    # Knn parameters
    params['nb_knn'] = [1,2,3,4,5,6,7,8]
    params['temperature'] = 0.07
    # Dataset parameters
    params['normalization'] = [-9.,4.] # sample real : [-14.,6.] for sample synth : [-14.,3.], mstar : [-9.,4.] 
    params['eval_path'] = './data/mstar/test'
    params['labeled_path'] = './data/mstar/train'
    params['n_label'] = -1 # number of labeled data, '-1' to take all images
    # Network parameters
    params['no_network'] = False # no network for feature extraction
    params['resize_dim'] # if no network is used all images are croped in the center with the following rectangle size
    params['device'] = 'cuda'
    params['model_path'] = './pipeline/out/encoder_1ch'
    params['arch'] = 'vit_tiny'
    params['patch_size'] = 8
    params['patch_drop'] = 0
    # Additional data reduction
    params['reduction'] = True # Do a data reduction on top of the extracted features
    params['red_dim'] = 50 # number of components of the pca
    params['red_type'] = 'pca' # 'pca' or 'umap'
    # Path and name of the logger
    params['logger_path'] = './pipeline/out/logger_knn/mstar_encoder_1ch.txt'
    
    eval_knn(params)    