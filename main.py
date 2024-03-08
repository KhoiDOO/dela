import os, sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DELA Monitor')

    # DATASET
    parser.add_argument('--ds', type=str, default='mnist', choices = ['mnist'],
        help='dataset used in training')
    parser.add_argument('--bs', type=int, default=64,
        help='batch size used for data set')
    parser.add_argument('--pinmem', action='store_true',
        help='toggle to pin memory in data loader')
    parser.add_argument('--wk', type=int, default=12,  
        help='number of worker processor contributing to data preprocessing')

    # TRAINING GENERAL SETTINGS
    parser.add_argument('--didx', type=int, default=0,
        help='device index used in training')
    parser.add_argument('--seed', type=int, default=0,
        help='seed used in training')
    parser.add_argument('--method', type=str, default='ae', choices=['ae', 'vae'],
        help='method used in training')
    parser.add_argument('--epoch', type=int, default=1,
        help='number of epochs used in training')
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate')
    parser.add_argument('--trm', type=str, default='sup', choices = ['sup'],
        help='training mode')
    parser.add_argument('--ldim', type=int, default=16,
        help='size of latent representation')

    # LOGGING
    parser.add_argument('--wandb', action='store_true',
        help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true',
        help='toggle to use tensorboard for offline saving')
    parser.add_argument('--wandb_prj', type=str, default="DELA",
        help='toggle to use wandb for online saving')
    parser.add_argument('--wandb_entity', type=str, default="truelove",
        help='toggle to use wandb for online saving')
    
    args = parser.parse_args()

    if args.trm == 'sup':
        from trainer import SupervisedTrainer
        trainer_interface = SupervisedTrainer(args=args)
    else:
        raise Exception(f'The training mode {args.trm} is currently not supported')
    
    trainer_interface.run()