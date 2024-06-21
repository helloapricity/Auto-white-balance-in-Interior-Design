import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import logging
import torch
import torch.nn.functional as F
from typing import Optional, Any
from torchmetrics import MeanMetric
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

from models.wb_net import WBNet
from data.dataset import setup_dataset
from arguments import get_args
from utils.ops import get_sobel_kernel
from lightning.pytorch.loggers import WandbLogger

logger = logging.getLogger("__name__")


class LitAWB(LightningModule):
    def __init__(
        self, 
        model, 
        x_kernel,
        y_kernel,
        lr:float=0.01, 
        smooth_weight:int=1,
        dist:bool=False
    ):
        super().__init__()
        
        self.model = model
        self.smooth_weight = smooth_weight
        self.x_kernel = x_kernel
        self.y_kernel = y_kernel
        self.lr = lr
        self.sync_dist = True if dist else False
        self.mean_valid_loss = MeanMetric()

        
    def forward(self, x:torch.tensor):
        logits = self.model(x)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch[0], batch[1]
        rec_loss, smooth_loss = 0, 0
        for c in range(inputs.shape[1]):
            patch = inputs[:, c, :, :]
            gt_patch = targets[:, c, :, :, :]
            pred, pred_weights = self(patch)
            
            # calculate loss
            rec_loss += F.mse_loss(pred, gt_patch)
            
            # smooth loss
            smooth_loss += self.smooth_weight * (
                torch.sum(F.conv2d(pred_weights, self.x_kernel.to(pred_weights.device))) + torch.sum(F.conv2d(pred_weights, self.y_kernel.to(pred_weights.device)))
            )
        
        loss = (rec_loss / inputs.shape[0]) + (smooth_loss / inputs.shape[0])
        
        self.log("train/loss", loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)        
        self.log("train/rec_loss", rec_loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log("train/smooth_loss", smooth_loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch[0], batch[1]

        with torch.no_grad():
            pred, _ = self(inputs[:, 0, :, :])
            
        val_loss = F.mse_loss(pred,  targets[:, 0, :, :, :])
        
        self.mean_valid_loss.update(val_loss, weight=inputs.shape[0])
    
    def on_validation_epoch_end(self):
        self.log("val/loss", self.mean_valid_loss, prog_bar=True, sync_dist=self.sync_dist, logger=True)
        
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer =  torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-4)
        
        return [optimizer]

    def save_checkpoint(self, filepath, weights_only:bool=False, storage_options:Optional[Any]=None) -> None:
        checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
        self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
        self.strategy.barrier("Trainer.save_checkpoint")


def main(args):
    
    wandb_logger = WandbLogger(project=args.project_name, log_model="all")

    # load model
    wb_model = WBNet(
        norm=args.norm, 
        inchnls= 3 * len(args.wb_settings)
    )
    
    dist = True if len(args.device) > 1 else False
        
    x_kernel, y_kernel = get_sobel_kernel(chnls=len(args.wb_settings))
    litmodel = LitAWB(model= wb_model, lr=args.lr, smooth_weight=args.smoothness_weight,
                    x_kernel=x_kernel, y_kernel=y_kernel, dist=dist)
    
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
        
        # Saves only on the main process    
        saved_ckpt_path = f'{saved_ckpt_path}/checkpoint'
        os.makedirs(saved_ckpt_path, exist_ok=True)
        saved_ckpt_path = f'{saved_ckpt_path}/best.pt'
        trainer.save_checkpoint(saved_ckpt_path)
        
    if args.do_eval:
        logger.info("\n\n*** Evaluate ***")
        trainer.devices = 0
        trainer.test(litmodel, dataloaders=test_dataloader, ckpt_path="best")
        
        
if __name__ == '__main__':
    opt = get_args()
        
    # trainer
    logger.info('*** Start Training mode ***')
    main(opt)