import os, sys
import torch
from typing import *
import argparse
from datetime import datetime


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        self.prepare_device()
        self.prepare_save_dir()
        self.prepare_ds()
        self.prepare_wandb()
        self.prepare_tensorboard()
    
    def prepare_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda", index = self.args.didx)
        else:
            self.device = torch.device("cpu")

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
        pass

    def prepare_wandb(self):
        pass

    def prepare_tensorboard(self):
        pass

    def log(self, log_dict: Dict[str, int | float], key: str = None):
        pass
    
    def log_model(self):
        pass
    
    def train(self):
        not NotImplementedError()