import time
from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from ..core import PipelineError
from ..logger import LOGGER
from ..storage.state import StateStorageBase
from ..scheduler.base import SchedulerBase
from ..utils import move_to_device, save_model, load_model
from ..metrics.base import MetricsCalculatorBase
import os

tracker = True
w_value = 0.8

class TrainerSegmentation:
    def __init__(
            self,
            model: nn.Module,
            decoder: nn.Module,
            train_data_loader: Iterable,
            eval_data_loader: Iterable,
            epoch_count: int,
            opt: Optimizer,
            scheduler: SchedulerBase,
            loss: nn.Module,
            print_frequency: None or int,
            device: str,
            model_save_path: str,
            state_storage: StateStorageBase,
            tracker_storage: StateStorageBase,
            n_last_blocks: int,
            avgpool: bool,
            n_labels: int,
            metrics_calculator: MetricsCalculatorBase,) -> None:

        self.model = model.to(device)
        self.model = self.model.eval()
        self.decoder = decoder.to(device)
        self.loss = loss.to(device)
        self.opt = opt
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.epoch_count = epoch_count
        self.scheduler = scheduler
        self.print_frequency = print_frequency
        self.model_save_path = model_save_path
        self.state_storage = state_storage
        self.device = device
        self.n_last_blocks = n_last_blocks
        self.avgpool = avgpool
        self.n_labels = n_labels
        self.metrics_calculator = metrics_calculator
        if tracker:
            self.tracker_storage = tracker_storage
            self.loss_ce = []
            self.lr = []
            self.eval = []

    def log_train_step(self, epoch_id: int, step_id: int, epoch_time: float, loss: float, mean_loss: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{:.4} s] Epoch {}. Train step {}. Loss ce {:.4}. Mean loss {:.4}".format(
                epoch_time, epoch_id, step_id, loss, mean_loss))
            return True
        return False
    
    def filterData(self,tensor:torch.tensor,label:torch.tensor):
        assert label.shape == tensor.shape
        b,c,h,w = tensor.shape
        label,tensor = label.permute(0,2,3,1).reshape(-1,c),tensor.permute(0,2,3,1).reshape(-1,c)
        not_nan_indices = ~torch.isnan(label)
        not_nan_indices = torch.all(not_nan_indices,dim=1)
        filtered_tensor = tensor[not_nan_indices]
        filtered_label = label[not_nan_indices]
        return filtered_tensor,filtered_label

    def feature_extraction(self,x:torch.Tensor,patch_size = 32,stride = 32):
        b,c,h,w = x.shape
        # Number of patches along each dimension
        num_patches_H = h // stride
        num_patches_W = w // stride
        # Total length required to fit all patches exactly
        total_length_H = patch_size + (num_patches_H - 1) * stride
        total_length_W = patch_size + (num_patches_W - 1) * stride
        # Padding needed: difference between required total length and original dimension
        pad_h = total_length_H - h
        pad_w = total_length_W - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        # Slicing the image in non overlaping patches
        x = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        x = x.contiguous().view(b, c, -1, patch_size, patch_size)
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, c, patch_size, patch_size)
        # Extracing features
        intermediate_output = self.model.get_intermediate_layers(x, self.n_last_blocks)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if self.avgpool:
            output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            output = output.reshape(output.shape[0], -1)
        # Resizing to keep the spatial dimension
        output_h = num_patches_H
        output_w = num_patches_W
        output = output.reshape(b, output_h, output_w, output.shape[-1])
        output = output.permute(0, 3, 1, 2)
        return output
        

    def train_step(self, input_data: torch.Tensor, label: torch.Tensor):
        self.opt.zero_grad()
        # Compute training loss
        input_data = move_to_device(input_data, device=self.device)
        
        feats_16 = self.feature_extraction(input_data,patch_size=16,stride=32)
        feats_32 = self.feature_extraction(input_data,patch_size=32,stride=32)
        feats_64 = self.feature_extraction(input_data,patch_size=64,stride=32)

        output = self.decoder(feats_16,feats_32,feats_64)
        label = move_to_device(label, device=self.device)
        
        output,label = self.filterData(output,label)
        loss = self.loss(output.view(-1,self.n_labels), label.view(-1,self.n_labels)) 
        loss.backward()
       
        self.opt.step()
        loss = loss.cpu().data.numpy()
        if tracker:
            self.loss_ce.append(loss)
            self.lr.append(self.opt.param_groups[0]["lr"])
        return loss

    def log_train_epoch(self, epoch_id: int, epoch_time: float, mean_loss: float):
        LOGGER.info("Training Epoch {} has completed. Time: {:.6}. Mean loss: {:.6}".format(
            epoch_id, epoch_time, mean_loss))
        return True

    def run_train_epoch(self, epoch_id: int):
        self.decoder.train()
        start_time = time.time()
        mean_loss = 0
        step_count = 0

        for step_id, (input_data,label,_) in enumerate(self.train_data_loader):
            loss = self.train_step(input_data,label)
            epoch_time = time.time() - start_time
            mean_loss += loss
            step_count += 1
            self.log_train_step(epoch_id, step_id, epoch_time, loss, mean_loss / step_count)

        epoch_time = time.time() - start_time
        mean_loss /= max(step_count, 1)
        self.log_train_epoch(epoch_id, epoch_time, mean_loss)
            
    def predict_step(self ,input_data: torch.Tensor, label: torch.Tensor):
        input_data = move_to_device(input_data, device=self.device)

        feats_16 = self.feature_extraction(input_data,patch_size=16,stride=32)
        feats_32 = self.feature_extraction(input_data,patch_size=32,stride=32)
        feats_64 = self.feature_extraction(input_data,patch_size=64,stride=32)

        output = self.decoder(feats_16,feats_32,feats_64)
        label = move_to_device(label, device=self.device)

        output,label = self.filterData(output,label)
        loss = self.loss(output.view(-1,self.n_labels), label.view(-1,self.n_labels)) 
        
        self.metrics_calculator.add(output,label)
        return loss.cpu().data.numpy()


    def log_evaluation_epoch(self, epoch_id: int, epoch_time: float, mean_loss: float, metrics: dict):
        LOGGER.info("Evaluation Epoch {} has completed. Time: {:.6}. Mean loss: {:.6}. Metrics: {}".format(
            epoch_id, epoch_time, mean_loss, metrics))
        return True

    def run_evaluation_epoch(self,epoch_id:int):
        self.decoder.eval()
        self.metrics_calculator.zero_cache()
        mean_loss = 0
        step_count = 0
        start_time = time.time()
        with torch.no_grad():
            for step_id, (input_data,label,_) in enumerate(self.eval_data_loader):
                loss = self.predict_step(input_data,label)
                step_count += 1
                mean_loss += loss
                epoch_time = time.time() - start_time

        epoch_time = time.time() - start_time
        mean_loss /= max(step_count, 1)
        metrics = self.metrics_calculator.calculate()
        if tracker:
                self.eval.append(metrics)
        self.log_evaluation_epoch(epoch_id, epoch_time, mean_loss, metrics)

        return epoch_time, mean_loss, metrics 
     
   
    def load_optimizer(self):
        if not self.state_storage.has_key("opt"):
            print("optimizer state not found...")
            return
        self.opt.load_state_dict(self.state_storage.get_value("opt"))
        
    def save_optimizer(self):
        self.state_storage.set_value("opt", self.opt.state_dict())

    def save_last_model(self, epoch_id):
        LOGGER.info("Saving segmentation head for epoch {} in {}".format(epoch_id,self.model_save_path))
        model_path = os.path.join(self.model_save_path, "segmentation_epoch_{}".format(epoch_id))
        save_model(self.decoder, model_path)
        
    def load_last_model(self, epoch_id):
        last_model_path = os.path.join(self.model_save_path, "segmentation_epoch_{}".format(epoch_id))
        load_model(self.decoder, last_model_path)   

    def load_tracker(self):
        if not (self.tracker_storage.has_key('loss') and self.tracker_storage.has_key('lr') and self.tracker_storage.has_key('wd')):
            print('One or more tracker not found ...')
            return
        self.loss_ce = self.tracker_storage.get_value('loss cross entropy')
        self.lr = self.tracker_storage.get_value('lr')
        self.eval = self.tracker_storage.get_value('eval')

    def save_tracker(self):
        if not self.tracker_storage.has_key('n_step'):
            self.tracker_storage.set_value('n_step',len(self.train_data_loader))
        self.tracker_storage.set_value('loss cross entropy',self.loss_ce) 
        self.tracker_storage.set_value('lr',self.lr) 
        self.tracker_storage.set_value('eval',self.eval) 
            
    def run(self):
        if tracker:
            self.load_tracker()        
        # Load model and optimizer
        start_epoch_id = 0
        if self.state_storage.has_key("start_epoch_id"):
            last_saved_epoch = self.state_storage.get_value("start_epoch_id")-1
            start_epoch_id = last_saved_epoch+1
            try:
                self.load_last_model(last_saved_epoch)
                LOGGER.info("Last saved weights epoch {}, starting training epoch {}".format(last_saved_epoch,start_epoch_id))            
            except:
                LOGGER.exception("Exception occurs during loading a model. Starting to train a model from scratch...")  
        else:
            LOGGER.info("Model not found in {}. Starting to train a model from scratch...".format(self.model_save_path))
        self.load_optimizer()
        
        epoch_id = start_epoch_id
        while self.epoch_count is None or epoch_id < self.epoch_count:
        
            self.run_train_epoch(epoch_id)
            self.run_evaluation_epoch(epoch_id)
            self.scheduler.step()

            if epoch_id % 10 == 0 or epoch_id> self.epoch_count-11: 
                self.save_optimizer()
                self.save_last_model(epoch_id)
                self.state_storage.set_value("start_epoch_id", epoch_id + 1)
                if tracker:
                    self.save_tracker()
            epoch_id += 1

            
















