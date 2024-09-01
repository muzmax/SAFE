import os

from configs.train_config_base import ConfigContrastive

from torchvision import transforms
from torchvision import models as torchvision_models
from torch.nn import SyncBatchNorm
import torch.optim

import pipeline.models.ViT as vits
from pipeline.datasets.load import train_data
from pipeline.datasets.datasets import train
from pipeline.datasets.preprocessing import *
from pipeline.models.ViT import ViTHead
from pipeline.loss.proto import ProtoLoss
from pipeline.trainer.trainer_Safe import TrainerSafe
from pipeline.scheduler.cosine_safe import Scheduler_safe
from pipeline.storage.state import StateStorageFile
from pipeline.utils import LARS, get_params_groups
from pipeline.logger import LOGGER

class Config_Safe(ConfigContrastive):
    def __init__(self,params):
        device = params['device']
        save_dir = params['data']['save_dir']
        batch_size = params['data']['batch_size']
        arch = params['model']['model_name']
        num_proto = params['model']['num_proto']
        out_dim = params['model']['output_dim']
        epochs = params['optimization']['epochs']
        opt = params['optimization']['optimizer']
        # ============ creating storage and Trainer ... ============
        state_storage = StateStorageFile(os.path.join(save_dir, "state"))
        tracker_storage = StateStorageFile(os.path.join(save_dir, "tracker"))
        trainer = TrainerSafe
        # ============ creating datasets and transformations ... ============
        dataset = train_data(
                            params['data']['train_dir'],
                            pat_size=params['data']['patch_size'],
                            stride=params['data']['stride'],
                            bat_size=batch_size,
                            nb_ch=params['model']['channels'])

        process = DataAugmentationSAR(global_crops_scale=params['data']['global_crop_size'],
                                    local_crops_scale=params['data']['local_crop_size'],
                                    subres_crop_scale=params['data']['local_crop_size'],
                                    global_crops_number=params['data']['global_views'],
                                    local_crops_number=params['data']['local_views'],
                                    subres_crop= params['data']['subres_crop'],
                                    shift_proba=params['data']['shift_param'][0],
                                    shift_min=params['data']['shift_param'][1],
                                    shift_max=params['data']['shift_param'][2],
                                    device=device)
        train_patches = train(dataset,process)
        len_train_patches = int(len(train_patches)/batch_size)+1
        # ============ building Loss ... ============
        loss = ProtoLoss(
        nproto= num_proto,
        warmup_teacher_temp = params['optimization']['teacher_temperature'],
        teacher_temp = params['optimization']['teacher_temperature'],
        warmup_teacher_temp_epochs = 0,
        nepochs = epochs,
        device=device)
        # ============ building student and teacher networks ... ============
        # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
        if arch in vits.__dict__.keys():
            student = vits.__dict__[arch](
                patch_size = params['model']['patch_size'],
                in_chans = params['model']['channels'],
                drop_path_rate=params['model']['drop_path_rate'])  
            student.set_patch_drop(params['model']['patch_drop'])
            teacher = vits.__dict__[arch](
                patch_size = params['model']['patch_size'],
                in_chans = params['model']['channels'])
            embed_dim = student.embed_dim
        # otherwise, we check if the architecture is in torchvision models
        elif arch in torchvision_models.__dict__.keys():
            student = torchvision_models.__dict__[arch]()
            teacher = torchvision_models.__dict__[arch]()
            embed_dim = student.fc.weight.shape[1]
        else:
            print(f"Unknow architecture: {arch}")

        head_student = ViTHead(
                                in_dim = embed_dim,
                                out_dim = out_dim,
                                use_bn = params['model']['use_bn'],
                                norm_last_layer = params['model']['use_norm'])
        head_teacher = ViTHead(
                                in_dim = embed_dim,
                                out_dim = out_dim,
                                use_bn = params['model']['use_bn'])

        teacher.load_state_dict(student.state_dict())
        head_teacher.load_state_dict(head_student.state_dict())

        for p in teacher.parameters():
            p.requires_grad = False
        for p in head_teacher.parameters():
            p.requires_grad = False

        LOGGER.info(f"Student and Teacher are built: they are both {arch} network.")
        # ============ building prototypes ... ============
        assert num_proto > 0
        with torch.no_grad():
            prototypes = torch.empty(num_proto, out_dim)
            _sqrt_k = (1./out_dim)**0.5
            torch.nn.init.uniform_(prototypes, -_sqrt_k, _sqrt_k)
            prototypes = torch.nn.parameter.Parameter(prototypes)
        if not params['model']['freeze_proto']:
            prototypes.requires_grad = True
        LOGGER.info('Created prototypes: {} - Requires grad: {}'.format(prototypes.shape,prototypes.requires_grad))
        # ============ preparing optimizer ... ============
        params_groups = get_params_groups([student,head_student],prototypes=prototypes,lr_prototypes=1e-3)
        if opt == "adamw":
            optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        elif opt == "sgd":
            optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        elif opt == "lars":
            optimizer = LARS(params_groups)  # to use with convnet and large batches
        # ============ preparing schedulers ... ============
        LR_ENC= params['optimization']['encoder']['lr']
        param_lr = {
        'base_value' : LR_ENC*batch_size/ 256.,
        'final_value' : params['optimization']['encoder']['final_lr'],
        'epoch' : epochs,
        'niter_epoch' : len_train_patches,
        'warmup_epochs' : params['optimization']['encoder']['warmup_epochs']}
        param_wd = {
        'base_value' :  params['optimization']['weight_decay']['base_value'],
        'final_value' : params['optimization']['weight_decay']['final_value'],
        'epoch' : epochs,
        'niter_epoch' : len_train_patches}
        param_momentum = {
        'base_value' : params['optimization']['momentum']['base_value'],
        'final_value' : params['optimization']['momentum']['final_value'],
        'epoch' : epochs,
        'niter_epoch' : len_train_patches}
        scheduler = Scheduler_dino(param_lr,param_wd,param_momentum) 
        # ============ init ... ============
        super().__init__(model_save_path=save_dir,
                        train_dataset=train_patches,
                        trainer=trainer,
                        encoder_1=student,
                        encoder_2=teacher,
                        proj_1=head_student,
                        proj_2=head_teacher,
                        opt=optimizer,
                        loss=loss,
                        device=device,
                        scheduler=scheduler,
                        batch_size=batch_size,
                        num_workers=params['data']['num_workers'],
                        epoch_count=epochs,
                        print_frequency=params['print_frequency'],
                        state_storage=state_storage,
                        tracker_storage = tracker_storage,
                        prototypes=prototypes)


