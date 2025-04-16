import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import logging
# import torch
# import torch.nn.functional as F
# from typing import Optional, Any
# from torchmetrics import MeanMetric
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from models.wb_net import WBNet
# from models.litawb_module import LitAWB
from models.litawb_style_module import LitAWBStyleLoss
from models.vgg19 import vgg19_net
from data.dataset import setup_dataset
from arguments import get_args
from utils.ops import get_sobel_kernel
from lightning.pytorch.loggers import WandbLogger

logger = logging.getLogger("__name__")

os.system("wandb login --relogin 3b704d1e75ac5487b451b58762bab4280d03b7c7")

def main(args):
    
    wandb_logger = WandbLogger(project=args.project_name, log_model="all")

    # load model
    wb_model = WBNet(
        norm=args.norm, 
        inchnls= 3 * len(args.wb_settings)
    )
    

    dist = True if len(args.device) > 1 else False
        
    x_kernel, y_kernel = get_sobel_kernel(chnls=len(args.wb_settings))
    # litmodel = LitAWB(model= wb_model, lr=args.lr, smooth_weight=args.smoothness_weight,
    #                 x_kernel=x_kernel, y_kernel=y_kernel, dist=dist)
    
    litmodel = LitAWBStyleLoss(model= wb_model,
                               lr=args.lr,
                               smooth_weight=args.smoothness_weight,
                               x_kernel=x_kernel,
                               y_kernel=y_kernel,
                               dist=dist,
                               vgg_model=vgg19_net)
    
    # load_dataset
    if args.do_train:
        train_dataloader = setup_dataset(
            imgfolders=args.training_dir,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            patch_number=args.patch_number,
            aug=args.aug,
            mode='training',
            multiscale=args.multiscale,
            keep_aspect_ratio=args.keep_aspect_ratio,
            t_size=args.img_size,
            num_workers=args.num_workers
        )
    
    if args.do_eval:
        test_dataloader = setup_dataset(
            imgfolders=args.valdir,
            batch_size=args.batch_size * 2,
            patch_size=args.patch_size,
            patch_number=1,
            aug=False,
            mode='validation',
            multiscale=False,
            keep_aspect_ratio=False,
            t_size=args.img_size,
            num_workers=args.num_workers
        )
    
    # create callback functions
    model_checkpoint = ModelCheckpoint(
                        save_top_k=3,
                        monitor="val/loss",
                        mode="min", dirpath=args.output_path,
                        filename="sample-{epoch:02d}",
                        save_weights_only=True)
    
    # create Trainer
    trainer = Trainer(
        max_epochs=args.epochs, 
        accelerator=args.accelerator, 
        devices=args.device, 
        callbacks=[model_checkpoint], 
        strategy='fsdp' if dist else 'auto',
        log_every_n_steps=1,
        logger=wandb_logger
    )
    
    if args.do_train:
        logger.info("*** Start training ***")
        trainer.fit(
            model=litmodel, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=test_dataloader if args.do_eval else None
        )
        
    if args.do_eval:
        logger.info("\n\n*** Evaluate ***")
        trainer.devices = 0
        trainer.test(litmodel, dataloaders=test_dataloader, ckpt_path="best")

        
if __name__ == '__main__':
    opt = get_args()
        
    # trainer
    logger.info('*** Start Training mode ***')
    main(opt)