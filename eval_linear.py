import os

import torch
from torch import nn
from torchvision import transforms 
from torch.utils.data import DataLoader

import pipeline.models.ViT as vits
from pipeline.utils import load_model
from pipeline.scheduler.base import SchedulerWrapperIdentity
from pipeline.trainer.trainer_linear import TrainerLinear
from pipeline.storage.state import StateStorageFile
from pipeline.logger import setup_logger,LOGGER
from pipeline.loss.SoftmaxSmooth import SoftmaxSmooth
from pipeline.metrics.accuracy import MetricsCalculatorAccuracy
from pipeline.datasets.load import load_separated_paths, load_paths
from pipeline.datasets.preprocessing import normalization,ToTensor,shift
from pipeline.datasets.datasets import data_classification, multisize_collate_fn

def train_linear(params):
    # ============ building datasets ... ============
    # try center crop, bias, batch size and optimizer
    transform_train = transforms.Compose([normalization(params['normalization'][0],params['normalization'][1]),ToTensor()])
    transform_eval = transforms.Compose([normalization(params['normalization'][0],params['normalization'][1]),ToTensor()])
    trainpaths, evalpaths, classes = load_separated_paths(params['train_path'],
                                                          params['eval_path'],
                                                          n_label=params['n_label'],
                                                          batch_sz=params['batch_size'])
    
    params['batch_size'] = min(len(trainpaths),params['batch_size'])
    
    train_data = data_classification(trainpaths,transform_train,classes)
    eval_data = data_classification(evalpaths,transform_eval,classes)

    train_dataloader = DataLoader(train_data,
                                  batch_size=params['batch_size'], 
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=multisize_collate_fn)
    eval_dataloader = DataLoader(eval_data,
                                  batch_size=1, 
                                  shuffle=False,
                                  num_workers=0)
    # ============ building logger ... ============
    os.makedirs(params['save_path'],exist_ok=True)
    logger_path = os.path.join(params['save_path'],"log_train.txt")
    setup_logger(out_file=logger_path,setup_msg=False)
    # ============ building backbone ... ============
    if params['arch'] in vits.__dict__.keys():
            model = vits.__dict__[params['arch']](patch_size=params['patch_size'])
            model.set_patch_drop(params['patch_drop'])
            model.to(params['device'])
            load_model(model, params['model_path'])
            model.eval()
            embed_dim = model.embed_dim * (params['n_last_blocks'] + int(params['avgpool_patchtokens']))
    else :
        print("Unknow architecture: {}".format(params['arch']))
    # ============ building linear layer and training parameters ... ============
    linear_classifier = LinearClassifier(embed_dim, num_labels=params['num_labels'])
    linear_classifier.to(params['device'])

    # optimizer = torch.optim.SGD(
    #     linear_classifier.parameters(),
    #     params['lr'] * (params['batch_size']) / 256., # linear scaling rule
    #     momentum=0.9,
    #     weight_decay=0)# we do not apply weight decay
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, params['epochs'], eta_min=0)
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=params['lr'], weight_decay=0)
    scheduler = SchedulerWrapperIdentity()
    
    loss = torch.nn.CrossEntropyLoss()
    # loss = SoftmaxSmooth(n_labels=params['num_labels'],smoothing=0.1)
    # ============ initializing storage and metric ... ============
    state_storage = StateStorageFile(os.path.join(params['save_path'], "state"))
    tracker_storage = StateStorageFile(os.path.join(params['save_path'], "tracker"))
    metrics_calculator = MetricsCalculatorAccuracy()
    # ============ training ... ============
    train = TrainerLinear(model,
                          linear_classifier,
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
                          metrics_calculator)
    train.run()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # linear layer
        return self.linear(x)
    
if __name__ == '__main__':

    params = {}
    # Save path for weights and logger
    params['save_path'] = './pipeline/out/linear/model_1' # model is saved every 10 epochs and at the last one
    # Dataset parameters
    params['normalization'] = [-9.,4.]  # mstar : [-9.,4.] 
    params['eval_path'] = './data/mstar/test'
    params['train_path'] = './data/mstar/train'
    params['n_label'] = 5
    # Backbone parameters
    params['device'] = 'cuda'
    params['model_path'] = 'pipeline/out/encoder_1ch'
    params['arch'] = 'vit_tiny'
    params['patch_size'] = 8
    params['patch_drop'] = 0
    # Linear parameters
    params['n_last_blocks'] = 5
    params['avgpool_patchtokens'] = False
    params['num_labels'] = 10
    # Training parameters
    params['lr'] = 0.001
    params['batch_size'] = 128
    params['epochs'] = 500
    params['print_frequency'] = 100

    train_linear(params)

    