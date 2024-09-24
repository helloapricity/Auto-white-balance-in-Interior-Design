import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import logging
import torch
import torch.nn.functional as F
from typing import Optional, Any
from torchmetrics import MeanMetric
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

from models.wb_net_v3_StyleLoss_LCNorm import WBNet
from data.dataset import setup_dataset
from arguments import get_args
from utils.ops import get_sobel_kernel
from lightning.pytorch.loggers import WandbLogger
from utils import utils
from models.style_model import StyleModel
from utils.color_loss import COLORLoss

STYLE_WEIGHT = 1e2
# 2 STYLE_WEIGHT = 1e4
# 1 STYLE_WEIGHT = 1e5

# Batch Norm

logger = logging.getLogger("__name__")

loss_mse = torch.nn.MSELoss()
loss_color = COLORLoss()

# Calculate Gram matrix (G = FF^T)
def gram(x):
    device = torch.device('cuda:1')
    x = x.to(device)
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)  
    f_T = f.transpose(1, 2) 
    G = f.bmm(f_T) / (ch * h * w) 
    return G

# smooth_weight:int=1

class LitAWB(LightningModule):
    def __init__(
        self, 
        model, 
        x_kernel,
        y_kernel,
        lr:float=0.01, 
        smooth_weight:int=2,
        dist:bool=True,
        style_model=None
    ):
        super().__init__()
        
        self.model = model
        self.smooth_weight = smooth_weight
        self.x_kernel = x_kernel
        self.y_kernel = y_kernel
        self.lr = lr
        self.sync_dist = True if dist else False
        self.mean_valid_loss = MeanMetric()
        self.style_model = style_model
        
        # Check if model parameters are being registered
        # print(f"Number of parameters in the model: {sum(p.numel() for p in self.model.parameters())}")
        # print(f"Number of parameters in the style model: {sum(p.numel() for p in self.style_model.parameters())}")
        
    def forward(self, x:torch.tensor):
        logits = self.model(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch[0].float().to(self.device), batch[1].float().to(self.device)
        # print(batch[0].shape) inputs shape:   [32, 32, 9, 64, 64]
        # print(batch[1].shape) targets shape:  [32, 32, 3, 64, 64]
        
        rec_loss, smooth_loss, style_loss, color_loss = 0, 0, 0, 0
        
        for c in range(inputs.shape[1]):
            # inputs.shape[1] = 32
            patch = inputs[:, c, :, :].to(self.device)
            # print(patch.shape)          # [32, 9, 64, 64]     INPUT
            gt_patch = targets[:, c, :, :, :].to(self.device)
            # print(gt_patch.shape)       # [32, 3, 64, 64]     TARGET
            pred, pred_weights = self(patch)
            # print(pred.shape)           # [32, 3, 64, 64]     OUTPUT
            # print(pred_weights.shape)   # [32, 3, 64, 64]   

            # Calculate loss
            rec_loss += F.mse_loss(pred, gt_patch)
            color_loss += loss_color(pred, gt_patch)
            
            # Smooth loss
            smooth_loss += self.smooth_weight * (
                torch.sum(F.conv2d(pred_weights, self.x_kernel.to(pred_weights.device))) + torch.sum(F.conv2d(pred_weights, self.y_kernel.to(pred_weights.device)))
            )

            # Calculate gram matrices for style feature layer maps we care about
            style_features = self.style_model(gt_patch.to(self.device))
            style_gram = [gram(fmap) for fmap in style_features]

            # Get VGG features
            y_hat_features = self.style_model(pred.to(self.device))

            # Style loss
            y_hat_gram = [gram(fmap) for fmap in y_hat_features]
            # for idx, (sg, yg) in enumerate(zip(style_gram, y_hat_gram)):
                # print(f"Shape of style_gram[{idx}]: {sg.shape}")
                # print(f"Shape of y_hat_gram[{idx}]: {yg.shape}")
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:inputs.shape[0]])
                # print('DONE!')
            style_loss = STYLE_WEIGHT * style_loss
            
        # print("Flag")
        loss = (rec_loss / inputs.shape[0]) + (smooth_loss / inputs.shape[0]) + (style_loss / inputs.shape[0]) + (color_loss / inputs.shape[0])

        self.log("train/loss", loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log("train/rec_loss", rec_loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log("train/smooth_loss", smooth_loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log("train/style_loss", style_loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log("train/color_loss", color_loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        return loss

    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch[0].float(), batch[1].float()
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

def attem_load_author(model, checkpoint_path):
    # weights = torch.load(checkpoint_path)['state_dict']
    weights = torch.load(checkpoint_path)
    print(weights)
    print('-----------------')
    print(model)
    reweights = dict()
    for k, v in weights.items():
        flag = 1
        for key in ["model.epoch", "model.global_step", "model.pytorch-lightning_version", "model.state_dict", "model.loops"]:
            if key in k:
                flag = 0
        if flag == 0:
            continue       
        # reweights["model." + k] = v
        reweights[k] = v
        
    
    model.load_state_dict(reweights)
    # model.eval()
    
    return model

def main(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    wandb_logger = WandbLogger(project=args.project_name, log_model="all")

    # load wgg16 model
    # vgg16 = StyleModel().to(device)
    vgg16 = StyleModel().to("cuda:0")
    # print(vgg16)
    
    # load wb model
    wb_model = WBNet(
        norm=args.norm, 
        inchnls= 3 * len(args.wb_settings)
    ).to(device)
    
    dist = True if len(args.device) > 1 else False
        
    x_kernel, y_kernel = get_sobel_kernel(chnls=len(args.wb_settings))
    print(len(args.wb_settings))
    
    litmodel = LitAWB(model= wb_model, lr=args.lr, smooth_weight=args.smoothness_weight,
                    x_kernel=x_kernel, y_kernel=y_kernel, dist=dist, style_model=vgg16).to(device)
    
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
            batch_size=args.batch_size,
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
        accelerator="cuda",
        devices=[1],
        callbacks=[model_checkpoint], 
        accumulate_grad_batches=args.grad_acc,
        # name="train_full_indoor_3_images_1e-4",
        # strategy='fsdp' if dist else 'auto',
        strategy='ddp_find_unused_parameters_true' if dist else 'auto',
        # strategy = "auto",
        # strategy = 'ddp_find_unused_parameters_true',
        log_every_n_steps=9,
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
        # saved_ckpt_path = f'{saved_ckpt_path}/checkpoint'
        # os.makedirs(saved_ckpt_path, exist_ok=True)
        # saved_ckpt_path = f'{saved_ckpt_path}/best.pt'
        # trainer.save_checkpoint(saved_ckpt_path)
        
    if args.do_eval:
        logger.info("\n\n*** Evaluate ***")
        trainer.devices = 0
        trainer.test(litmodel, dataloaders=test_dataloader, ckpt_path="best")
        ''
        
if __name__ == '__main__':
    opt = get_args()
        
    # trainer
    logger.info('*** Start Training mode ***')
    main(opt)
    