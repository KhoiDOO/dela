from argparse import Namespace
from .core import Trainer, Algorithm

from rich.progress import track

import torch

class SupervisedTrainer(Trainer):
    def __init__(self, args: Namespace) -> Trainer:
        super().__init__(args)

    def run(self):
        for epoch in track(range(self.epoch)):

            train_log_dict = None
            valid_log_dict = None
            old_valid_loss = 1e26

            for x, y in self.train_dl:
                tr_loss, tr_log_dict = self.model(x=x)

                self.optimizer.zero_grad()
                tr_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if train_log_dict is None:
                    train_log_dict = {key : [tr_log_dict[key]] for key in tr_log_dict}
                else:
                    for key in train_log_dict:
                        train_log_dict[key].append(tr_log_dict[key])

            with torch.no_grad():
                for x, y in self.valid_dl:
                    tr_loss, vl_log_dict = self.model(x=x)

                    if valid_log_dict is None:
                        valid_log_dict = {key : [vl_log_dict[key]] for key in vl_log_dict}
                    else:
                        for key in valid_log_dict:
                            valid_log_dict[key].append(vl_log_dict[key])
                
            mean_train_log_dict = {key : sum(train_log_dict[key])/len(train_log_dict[key]) for key in train_log_dict}
            mean_valid_log_dict = {key : sum(valid_log_dict[key])/len(valid_log_dict[key]) for key in valid_log_dict}

            for dct in [mean_train_log_dict, mean_valid_log_dict]:
                self.log(log_dict=dct, epoch=epoch)
            
            mean_valid_loss = mean_valid_log_dict["valid/loss"]
            save_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': mean_valid_loss
            }

            if  mean_valid_loss <= old_valid_loss:
                old_valid_loss = mean_valid_loss

                save_path = self.args.save_dir + f"/best.pt"
                torch.save(save_dict, save_path)
            
            save_path = self.args.save_dir + f"/last.pt"
            torch.save(save_dict, save_path)

        self.log_model()