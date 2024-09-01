import os

import torch
from torch import nn
from torchvision import transforms 
from torch.utils.data import DataLoader

import pipeline.models.ViT as vits
from pipeline.models.segmentation import multi_feat_head
from pipeline.utils import load_model
from pipeline.trainer.trainer_segmentation import TrainerSegmentation
from pipeline.storage.state import StateStorageFile
from pipeline.logger import setup_logger,LOGGER
from pipeline.metrics.segmentation import MetricsCalculatorSegmentation
from pipeline.datasets.load import load_segmentation
from pipeline.datasets.preprocessing import SegmentationAugmentation
from pipeline.datasets.datasets import data_segmentation

def train_segmentation(params):
    # ============ building logger ... ============
    os.makedirs(params['save_path'],exist_ok=True)
    logger_path = os.path.join(params['save_path'],"log_train.txt")
    setup_logger(out_file=logger_path,setup_msg=False)
    # ============ building datasets ... ============
    transform_train = SegmentationAugmentation(params['normalization'],train=True)
    transform_pred = SegmentationAugmentation(params['normalization'],train=False)

    train_patches, train_labels = load_segmentation(params['train_path'],
                                               params['labels'],
                                               params['num_labels'],
                                               params['input_dim'],
                                               params['batch_size'],
                                               params['resize'])
    eval_patches, eval_labels = load_segmentation(params['eval_path'],
                                               params['labels'],
                                               params['num_labels'],
                                               params['input_dim'],
                                               params['batch_size'],
                                               params['resize'])
    train_data = data_segmentation(train_patches,train_labels,transform_train,params['num_labels'])
    eval_data = data_segmentation(eval_patches,eval_labels,transform_pred,params['num_labels'])
    train_dataloader = DataLoader(train_data,
                                  batch_size=params['batch_size'], 
                                  shuffle=True,
                                  num_workers=0)
    eval_dataloader = DataLoader(eval_data,
                                  batch_size=params['batch_size'], 
                                  shuffle=False,
                                  num_workers=0)
    # ============ building backbone ... ============
    if params['arch'] in vits.__dict__.keys():
            model = vits.__dict__[params['arch']](patch_size=params['patch_size'],
                                                  in_chans = params['n_channels'],)
            model.set_patch_drop(params['patch_drop'])
            model.to(params['device'])
            load_model(model, params['model_path'])
            model.eval()
            embed_dim = model.embed_dim * (params['n_last_blocks'] + int(params['avgpool_patchtokens']))
    else :
        print("Unknow architecture: {}".format(params['arch']))
    # ============ building segmentation head and training parameters ... ============
    decoder = multi_feat_head(sz_fmap = embed_dim,
                              n_upsample = 3,
                              n_class = params['num_labels'])

    decoder.to(params['device'])
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        params['lr']) # linear scaling ruley
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,60,80], gamma=0.1)
    loss = torch.nn.CrossEntropyLoss()
    # ============ initializing storage and metric ... ============
    state_storage = StateStorageFile(os.path.join(params['save_path'], "state"))
    tracker_storage = StateStorageFile(os.path.join(params['save_path'], "tracker"))
    metrics_calculator = MetricsCalculatorSegmentation(params['num_labels'])
    # ============ training ... ============
    train = TrainerSegmentation(model,
                                decoder,
                                train_dataloader,
                                eval_dataloader,
                                params['epochs'],
                                optimizer,
                                scheduler,
                                loss,
                                params['print_frequency'],
                                params['device'],
                                params['save_path'],
                                state_storage,
                                tracker_storage,
                                params['n_last_blocks'],
                                params['avgpool_patchtokens'],
                                params['num_labels'],
                                metrics_calculator)
    train.run()


if __name__ == '__main__':

    params = {}
    # Save path for weights and logger
    params['save_path'] = './pipeline/out/segmentation/example' # save model every 10 epochs and the last 10 epochs
    # Dataset parameters
    params['normalization'] = [-0.5,11.5] 
    params['train_path'] = './data/Raw_AIR-PolarSAR-Seg/train_set'
    params['eval_path'] = './data/Raw_AIR-PolarSAR-Seg/test_set'
    params['input_dim'] = 512
    params['resize'] = False
    # Backbone parameters
    params['device'] = 'cuda'
    params['model_path'] = 'pipeline/out/encoder_4ch'
    params['arch'] = 'vit_tiny'
    params['patch_size'] = 8
    params['patch_drop'] = 0
    params['n_channels'] = 4
    # Segmentation parameters
    params['n_last_blocks'] = 1
    params['avgpool_patchtokens'] = False
    params['num_labels'] = 6
    params['labels'] = {(0,0,255) : 0, # 'industrial'
                        (0,255,0) : 1, # 'natural'
                        (255,0,0) : 2, # 'land use'
                        (0,255,255) : 3, # 'water'
                        (255,255,0) : 4, # 'housing'
                        (255,255,255) : 5, # 'other'
                        (0,0,0) : None} # 'unlabeled', None values are not taken into account in the training

    # Training parameters
    params['lr'] = 0.001
    params['batch_size'] = 8
    params['epochs'] = 100
    params['print_frequency'] = 100

    train_segmentation(params)
