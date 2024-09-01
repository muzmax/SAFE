import time
from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
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

class TrainerLinear:
    def __init__(
            self,
            model: nn.Module,
            linear_classifier: nn.Module,
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
            metrics_calculator: MetricsCalculatorBase) -> None:

        self.model = model.to(device)
        self.model = self.model.eval()
        self.linear_classifier = linear_classifier.to(device)
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
    
    def train_step(self, input_data: torch.Tensor, label: torch.Tensor):
        # Compute training loss
        input_data = move_to_device(input_data, device=self.device)
        label = move_to_device(label, device=self.device)
        self.opt.zero_grad()
        
        if not isinstance(input_data, list):
            input_data = [input_data]

        for i,batch in enumerate(input_data):
            intermediate_output = self.model.get_intermediate_layers(batch, self.n_last_blocks)
            output_batch = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if self.avgpool:
                output_batch = torch.cat((output_batch.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output_batch = output.reshape(output_batch.shape[0], -1)
            if i == 0:
                output = output_batch
            else :
                output = torch.cat((output, output_batch))        
        output = self.linear_classifier(output)
        loss = self.loss(output, label) 
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
        self.linear_classifier.train()
        start_time = time.time()
        mean_loss = 0
        step_count = 0

        for step_id, (input_data,label,_,_) in enumerate(self.train_data_loader):
            loss = self.train_step(input_data,label)
            epoch_time = time.time() - start_time
            mean_loss += loss
            step_count += 1
            self.log_train_step(epoch_id, step_id, epoch_time, loss, mean_loss / step_count)

        epoch_time = time.time() - start_time
        mean_loss /= max(step_count, 1)
        self.log_train_epoch(epoch_id, epoch_time, mean_loss)
    
    def log_evaluation_step(self, epoch_id: int, step_id: int, epoch_time: float, loss: float, mean_loss: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{:.6} s] Epoch {}. Evaluation step {}. Loss {:.6}. Mean loss {:.6}".format(
                epoch_time, epoch_id, step_id, loss, mean_loss))
            
    def predict_step(self ,input_data: torch.Tensor, label: torch.Tensor):
        input_data = move_to_device(input_data, device=self.device)
        label = move_to_device(label, device=self.device)
        intermediate_output = self.model.get_intermediate_layers(input_data, self.n_last_blocks)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if self.avgpool:
            output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            output = output.reshape(output.shape[0], -1)
        output = self.linear_classifier(output)
        loss = self.loss(output, label)
        self.metrics_calculator.add(output,label)
        return loss.cpu().data.numpy()


    def log_evaluation_epoch(self, epoch_id: int, epoch_time: float, mean_loss: float, metrics: dict):
        LOGGER.info("Evaluation Epoch {} has completed. Time: {:.6}. Mean loss: {:.6}. Metrics: {}".format(
            epoch_id, epoch_time, mean_loss, str(metrics)))
        return True

    def run_evaluation_epoch(self,epoch_id:int):
        self.linear_classifier.eval()
        self.metrics_calculator.zero_cache()
        mean_loss = 0
        step_count = 0
        start_time = time.time()
        with torch.no_grad():
            for step_id, (input_data,label,_,_) in enumerate(self.eval_data_loader):
                loss = self.predict_step(input_data,label)
                step_count += 1
                mean_loss += loss
                epoch_time = time.time() - start_time

                # self.log_evaluation_step(epoch_id, step_id, epoch_time, loss, mean_loss/step_count)

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
        LOGGER.info("Saving linear classifier for epoch {} in {}".format(epoch_id,self.model_save_path))
        model_path = os.path.join(self.model_save_path, "linear_epoch_{}".format(epoch_id))
        save_model(self.linear_classifier, model_path)
        
    def load_last_model(self, epoch_id):
        last_model_path = os.path.join(self.model_save_path, "linear_epoch_{}".format(epoch_id))
        load_model(self.linear_classifier, last_model_path)   

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
            if epoch_id%10 == 0 or epoch_id ==self.epoch_count-1: # save every 10 epochs
                self.state_storage.set_value("start_epoch_id", epoch_id + 1)
                self.save_optimizer()
                self.save_last_model(epoch_id)
                self.run_evaluation_epoch(epoch_id)
                if tracker:
                    self.save_tracker()
            epoch_id += 1

            
















