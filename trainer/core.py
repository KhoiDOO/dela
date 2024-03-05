import os, sys
sys.path.append("/".join(os.getcwd().split("/")[:-1]))
import torch
from typing import *
import argparse
from datetime import datetime
import time

import wandb
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import json

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        self.prepare_device()
        self.prepare_seed()
        self.prepare_save_dir()
        self.prepare_ds()
        
        if self.args.wandb:
            self.prepare_wandb()
        if self.args.log:
            self.prepare_tensorboard()
        
        self.prepare_config()

        self.prepare_method()
        if self.args.wandb:
            self.watch_model(model=self.model)
        
        self.epoch = self.args.epoch
    
    def prepare_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda", index = self.args.didx)
        else:
            self.device = torch.device("cpu")
    
    def prepare_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True

        self.gen_func = torch.Generator().manual_seed(self.args.seed)

    def prepare_save_dir(self):
        main_dir = "/".join(os.getcwd().split("/")[:-1])

        run_dir = main_dir + "/runs"
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        
        self.method_dir = run_dir + f"/{self.args.method}"
        if not os.path.exists(self.method_dir):
            os.mkdir(self.method_dir)
        
        self.ds_dir = self.method_dir + f"/{self.args.ds}"
        if not os.path.exists(self.ds_dir):
            os.mkdir(self.ds_dir)
        
        self.args.current_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.args.save_dir = self.ds_dir + f"/{self.current_time}"
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

    def prepare_ds(self):
        if self.args.ds == 'mnist':
            from dataset import CusMNIST
        
            self.train_ds = CusMNIST(train = True)
            self.valid_ds = CusMNIST(train = False)
        else:
            raise Exception(f"dataset {self.args.ds} is currently not supported")

        self.train_dl = DataLoader(
            self.train_ds, batch_size=self.args.bs, shuffle=True, 
            pin_memory=self.args.pinmem, num_workers=self.args.wk, 
            generator=self.gen_func
        )
        
        self.valid_dl = DataLoader(
            self.valid_ds, batch_size=self.args.bs, shuffle=True, 
            pin_memory=self.args.pinmem, num_workers=self.args.wk, 
            generator=self.gen_func
        )

    def prepare_wandb(self):
        self.args.run_name = f"{self.args.method}__{self.args.ds}__{int(time.time())}"

        self.__run = wandb.init(
            project=self.args.wandb_prj,
            entity=self.args.wandb_entity,
            config=self.args,
            name=self.args.run_name,
            force=True
        )

    def prepare_tensorboard(self):
        self.__writer = SummaryWriter(self.args.save_dir)
        self.__writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )
    
    def prepare_config(self):
        config_dict = vars(self.args)
        self.save_json(config_dict, path=self.args.save_dir+'/config.json')

    def prepare_method(self):
        if self.args.method == 'ae':
            self.model = None
        else:
            raise Exception(f"the method {self.args.method} is currently not supported")

        self.optimizer = Adam(self.model, lr=self.args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max = len(self.train_dl)*self.args.epochs)
    
    def log(self, log_dict: Dict[str, int | float], key: str = None):
        pass
    
    def log_model(self):
        best_path = self.args.save_dir + f"/best.pt"
        if os.path.exists(best_path):
            self.__run.log_model(path=best_path, model_name=f"{self.args.run_name}-best-model")
        
        last_path = self.args.save_dir + f"/last.pt"
        if os.path.exists(last_path):
            self.__run.log_model(path=last_path, model_name=f"{self.args.run_name}-last-model")
    
    def watch_model(self, model):
        self.__run.watch(models=model, log='all', log_freq=len(self.train_dl), log_graph=True)
    
    @staticmethod
    def save_json(dct, path):
        with open(path, "w") as outfile:
            json.dump(dct, outfile)
    
    def train(self):
        not NotImplementedError()