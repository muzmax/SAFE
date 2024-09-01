from pipeline.storage.state import StateStorageFile
from pipeline.scheduler.base import SchedulerWrapperIdentity
from pipeline.datasets.base import EmptyDataset
import torch
import os


class ConfigContrastive:
    def __init__(
            self,
            model_save_path,
            train_dataset,
            trainer,
            encoder_1,
            encoder_2,
            proj_1,
            proj_2,
            opt,
            loss,
            prototypes=None,
            device=None,
            scheduler=None,
            batch_size=1,
            num_workers=0,
            epoch_count=None,
            print_frequency=1,
            state_storage=None,
            tracker_storage = None):

        if scheduler is None:
            scheduler = SchedulerWrapperIdentity()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if state_storage is None:
            state_storage = StateStorageFile(os.path.join(model_save_path, "state"))

        if tracker_storage is None:
            tracker_storage = StateStorageFile(os.path.join(model_save_path, "tracker"))

        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2
        self.proj_1 = proj_1
        self.proj_2 = proj_2
        self.opt = opt
        self.loss = loss
        self.model_save_path = model_save_path
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scheduler = scheduler
        self.epoch_count = epoch_count
        self.print_frequency = print_frequency
        self.trainer = trainer
        self.device = device
        self.state_storage = state_storage
        self.tracker_storage = tracker_storage
        self.prototypes = prototypes



