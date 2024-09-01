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
from ..datasets.preprocessing import denormalization_
from ..utils import move_to_device, save_model, load_model, disp_sar
import os

tracker = True
debug = False # display grad norm
w_clip = False # Gradient clipping for reconstruction loss
w_value = 0.8


# for disp debug
denorm = denormalization_()


class TrainerSafe:
    def __init__(
            self,
            encoder_1: nn.Module,
            encoder_2: nn.Module,
            proj_1: nn.Module,
            proj_2: nn.Module,
            train_data_loader: Iterable,
            epoch_count: int,
            opt: Optimizer,
            scheduler: SchedulerBase,
            loss: nn.Module,
            print_frequency: None or int,
            device: str,
            model_save_path: str,
            state_storage: StateStorageBase,
            tracker_storage: StateStorageBase,
            prototypes: torch.nn.parameter.Parameter) -> None:

        self.encoder_1 = encoder_1.to(device)
        self.encoder_2 = encoder_2.to(device)
        self.proj_1 = proj_1.to(device)
        self.proj_2 = proj_2.to(device)
        self.loss = loss.to(device)
        self.prototypes = prototypes.to(device)
        self.opt = opt
        self.train_data_loader = train_data_loader
        self.epoch_count = epoch_count
        self.scheduler = scheduler
        self.print_frequency = print_frequency
        self.model_save_path = model_save_path
        self.state_storage = state_storage
        self.device = device
        
        if debug:
            self.grad_enc = []
            self.grad_proj = []
            self.grad_enc_max = 0
            self.grad_proj_max = 0

        if tracker:
            self.tracker_storage = tracker_storage
            self.loss_ce = []
            self.loss_rg = []
            self.lr = []
            self.wd = []
    
    # Display a batch for debuging
    def disp_debug(self, input_data: torch.Tensor, param):
        im = torch.squeeze(input_data[0,:,:,:]).cpu().data.numpy()
        if len(im.shape) == 3 :# multi polarization image
            im = im.transpose(1,2,0)
        im_d = denorm(im,param[0][0].cpu().data.numpy(),param[0][1].cpu().data.numpy())
        print('\nbatch shape : {}'.format(input_data.shape))
        print('batch - mean {} - std {}'.format(torch.mean(input_data),torch.std(input_data)))
        print('im - mean {} - std {}'.format(np.mean(im),np.std(im)))
        print('im denorm - mean {} - std {}'.format(np.mean(im_d),np.std(im_d)))
        disp_sar(im_d,tresh=None)


    def train_step(self, input_data: torch.Tensor, epoch:int, step:int,param_norm):
        lr_step,wd_step,m_step = self.scheduler.step(len(self.train_data_loader)*epoch+step)

        # Update learning rate and weight decay
        for i, param_group in enumerate(self.opt.param_groups):
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_step
            if i != 2:  # prototypes lr are not updated with scheduler
                param_group["lr"] = lr_step

        # Compute training loss
        input_data = move_to_device(input_data, device=self.device)
        self.opt.zero_grad()
        out_student = self.proj_1(self.encoder_1(input_data[1:]))
        out_teacher = self.proj_2(self.encoder_2(input_data[:1]))
        loss_ce, loss_rg = self.loss(out_student, out_teacher, self.prototypes, epoch)
        loss = loss_ce+loss_rg
        loss.backward()
        # Gradient clipping
        if w_clip:
                nn.utils.clip_grad_value_(self.encoder_1.parameters(), clip_value=1.0)
                nn.utils.clip_grad_value_(self.proj_1.parameters(), clip_value=1.0)
                nn.utils.clip_grad_value_(self.prototypes, clip_value=1.0)
        # Gradient tracking
        if debug :
            for p in list(filter(lambda p: p.grad is not None, self.encoder_1.parameters())):
                    grads = p.grad.detach()
                    norm_grad = grads.data.norm().cpu().numpy()
                    max_norm_grad = torch.max(torch.abs(grads)).cpu().item()
                    self.grad_enc_max = max(self.grad_enc_max,max_norm_grad)
                    self.grad_enc.append(norm_grad)
            for p in list(filter(lambda p: p.grad is not None, self.proj_1.parameters())):
                    grads = p.grad.detach()
                    norm_grad = grads.data.norm().cpu().numpy()
                    max_norm_grad = torch.max(torch.abs(grads)).cpu().item()
                    self.grad_proj_max = max(self.grad_proj_max,max_norm_grad)
                    self.grad_proj.append(norm_grad)
        self.opt.step()
        loss_ce, loss_rg = loss_ce.cpu().data.numpy(), loss_rg.cpu().data.numpy()
        # Update teacher
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder_1.parameters(), self.encoder_2.parameters()):
                param_k.data.mul_(m_step).add_((1 - m_step) * param_q.detach().data)

        if tracker:
                self.loss_ce.append(loss_ce)
                self.loss_rg.append(loss_rg)
                self.lr.append(self.opt.param_groups[0]["lr"])
                self.wd.append(self.opt.param_groups[0]["weight_decay"])

        return loss_ce, loss_rg

    def log_train_step(self, epoch_id: int, step_id: int, epoch_time: float, loss_ce: float, loss_rg: float, mean_loss: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{:.4} s] Epoch {}. Train step {}. Loss regu {:.4}. Loss ce {:.4}. Mean loss {:.4}".format(
                epoch_time, epoch_id, step_id, loss_rg, loss_ce, mean_loss))
            if debug:
                print("\nAverage norm encoder : {}\nAverage norm projector : {}".format(np.mean(self.grad_enc),np.mean(self.grad_proj)))
                print("max norm encoder : {}\nmax norm projector : {}".format(np.mean(self.grad_enc_max),np.mean(self.grad_proj_max)))
            return True
        return False

    def log_train_epoch(self, epoch_id: int, epoch_time: float, mean_loss: float):
        LOGGER.info("Training Epoch {} has completed. Time: {:.6}. Mean loss: {:.6}".format(
            epoch_id, epoch_time, mean_loss))
        return True

    def run_train_epoch(self, epoch_id: int):
        self.encoder_1.train()
        self.encoder_2.train()
        self.proj_1.train()
        self.proj_2.train()
        start_time = time.time()
        mean_loss = 0
        step_count = 0

        for step_id, (input_data,param_norm) in enumerate(self.train_data_loader):
            loss_ce, loss_rg = self.train_step(input_data,epoch_id,step_id,param_norm)
            epoch_time = time.time() - start_time
            mean_loss += loss_ce
            step_count += 1
            self.log_train_step(epoch_id, step_id, epoch_time, loss_ce, loss_rg, mean_loss / step_count)

        epoch_time = time.time() - start_time
        mean_loss /= max(step_count, 1)
        self.log_train_epoch(epoch_id, epoch_time, mean_loss)

    def load_optimizer(self):
        if not self.state_storage.has_key("opt"):
            print("optimizer state not found...")
            return
        self.opt.load_state_dict(self.state_storage.get_value("opt"))
        
    def save_optimizer(self):
        self.state_storage.set_value("opt", self.opt.state_dict())
        

    def save_last_model(self, epoch_id):
        os.makedirs(self.model_save_path, exist_ok=True)
        index = ["encoder_1","encoder_2","proj_1","proj_2","prototypes"]
        LOGGER.info("Saving models for epoch {} in {}".format(epoch_id,self.model_save_path))
        for model in index :
            model_path = os.path.join(self.model_save_path, "{}_epoch_{}".format(model,epoch_id))
            if model == "encoder_1":
                save_model(self.encoder_1, model_path)
            elif model == "encoder_2":
                save_model(self.encoder_2, model_path)
            elif model == "proj_1":
                save_model(self.proj_1, model_path)
            elif model == "proj_2":
                save_model(self.proj_2, model_path)
            else:
                with open(model_path, "wb") as fout:
                    torch.save(self.prototypes, fout)
            

    def load_last_model(self, epoch_id):
        index = ["encoder_1","encoder_2","proj_1","proj_2","prototypes"]
        for model in index :
            last_model_path = os.path.join(self.model_save_path, "{}_epoch_{}".format(model,epoch_id))
            if model == "encoder_1":
                load_model(self.encoder_1, last_model_path)
            elif model == "encoder_2":
                load_model(self.encoder_2, last_model_path)
            elif model == "proj_1":
                load_model(self.proj_1, last_model_path)
            elif model == "proj_2":
                load_model(self.proj_2, last_model_path)
            else:
                with open(last_model_path, "rb") as fin:
                    self.prototypes = torch.load(fin)

    def load_tracker(self):
        if not (self.tracker_storage.has_key('loss') and self.tracker_storage.has_key('lr') and self.tracker_storage.has_key('wd')):
            print('One or more tracker not found ...')
            return
        self.loss_ce = self.tracker_storage.get_value('loss cross entropy')
        self.loss_rg = self.tracker_storage.get_value('loss mean regularization')
        self.lr = self.tracker_storage.get_value('lr')
        self.wd = self.tracker_storage.get_value('wd')

    def save_tracker(self):
        if not self.tracker_storage.has_key('n_step'):
            self.tracker_storage.set_value('n_step',len(self.train_data_loader))
        self.tracker_storage.set_value('loss cross entropy',self.loss_ce) 
        self.tracker_storage.set_value('loss mean regularization',self.loss_rg) 
        self.tracker_storage.set_value('lr',self.lr) 
        self.tracker_storage.set_value('wd',self.wd) 
       
    def run(self):
        if tracker:
            self.load_tracker()
        os.makedirs('./data/sample', exist_ok=True)
        
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

            if epoch_id%25 == 0 or epoch_id == self.epoch_count -1:
                self.state_storage.set_value("start_epoch_id", epoch_id + 1)
                self.save_optimizer()
                self.save_last_model(epoch_id)
                if tracker:
                    self.save_tracker()

            epoch_id += 1

            
















