from pipeline.scheduler.base import SchedulerBase
import numpy as np

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class Scheduler_safe(SchedulerBase):
    def __init__(self,param_lr,param_wd,param_mom) -> None:
        self.load(param_lr,param_wd,param_mom)
    
    def step(self,id):
        return self.learning_rate[id],self.weight_decay[id],self.momentum[id]
    
    def get_param(self):
        return self.learning_rate,self.weight_decay,self.momentum

    def load(self,param_lr,param_wd,param_mom):
        self.learning_rate = cosine_scheduler(param_lr['base_value'],
                                                param_lr['final_value'],
                                                param_lr['epoch'],
                                                param_lr['niter_epoch'],
                                                param_lr['warmup_epochs'])
                                                
        self.weight_decay = cosine_scheduler(param_wd['base_value'],
                                                param_wd['final_value'],
                                                param_wd['epoch'],
                                                param_wd['niter_epoch'])    
                                                                            
        self.momentum = cosine_scheduler(param_mom['base_value'],
                                                param_mom['final_value'],
                                                param_mom['epoch'],
                                                param_mom['niter_epoch'])

  
   