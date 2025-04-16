from lightning.pytorch import LightningModule
from torchmetrics import MeanMetric
from typing import Optional, Any
import torch.nn.functional as F
import torch
from models.vgg19 import get_features, layers, gram_matrix, gt_weights
from torchvision import models

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

class LitAWBStyleLoss(LightningModule):
    def __init__(
        self, 
        model, 
        vgg_model,
        x_kernel,
        y_kernel,
        lr:float=0.01, 
        smooth_weight:int=1,
        dist:bool=False
    ):
        super().__init__()
        
        self.model = model
        self.vgg_model = vgg_model
        self.smooth_weight = smooth_weight
        self.x_kernel = x_kernel
        self.y_kernel = y_kernel
        self.lr = lr
        self.sync_dist = True if dist else False
        self.mean_valid_loss = MeanMetric()
        
        # TODO: freeze vgg model
        for param in self.vgg_model.parameters():
            param.requires_grad_(False)
        self.vgg_model.to(device)
    
    def forward(self, x:torch.tensor):  
        logits = self.model(x)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch[0].float().to(device), batch[1].float().to(device)            # [32, 32, 9, 64, 64]
        # print(inputs.shape)
        style_loss = 0
        for c in range(inputs.shape[1]):
            patch = inputs[:, c, :, :]
            # print(patch.shape)                                        # [8, 9, 64, 64]
            gt_patch = targets[:, c, :, :, :]
            # print(gt_patch.shape)                                     # [8, 3, 64, 64]
            pred, pred_weights = self(patch)            
            # print(pred.shape)
            
            # calculate loss
            # TODO: calculate style loss
            for i in range(patch.shape[0]):
                gt_patch_i = gt_patch[i] 
                gt_patch_i = gt_patch_i.unsqueeze(0)
                gt_features = get_features(gt_patch_i, self.vgg_model, layers)
                # print(gt_patch_i.shape)                                         # [1, 3, 64, 64]
                
                gt_grams = {layer: gram_matrix(gt_features[layer]) for layer in gt_features}
                
                pred_patch_i = pred[i]
                pred_patch_i = pred_patch_i.unsqueeze(0)                            # [1, 3, 64, 64]
                target = pred_patch_i.clone().requires_grad_(True).to(device)
                # target = pred_patch_i.clone().requires_grad_(True)
                # print(pred_patch_i.shape)
                target_features = get_features(target, self.vgg_model, layers)
                for layer in gt_weights:
                    _, d, h, w = target_features[layer].shape
                    target_gram = gram_matrix(target_features[layer])
                    
                    layer_style_loss = gt_weights[layer] * torch.mean((target_gram - gt_grams[layer])**2)
                    style_loss += layer_style_loss / (d * h * w)
                # total_style_loss = style_weight * style_loss
                
        # TODO: add style loss
        loss = (style_loss / inputs.shape[0])
        
        self.log("train/loss", loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log("train/style_loss", style_loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch[0].float().to(device), batch[1].float().to(device)

        with torch.no_grad():
            pred, _ = self(inputs[:, 0, :, :])
            
        val_loss = F.mse_loss(pred,  targets[:, 0, :, :, :])
        
        self.mean_valid_loss.update(val_loss, weight=inputs.shape[0])
    
    def on_validation_epoch_end(self):
        self.log("val/loss", self.mean_valid_loss, prog_bar=True, sync_dist=self.sync_dist, logger=True)
        
    def test_step(self, batch, batch_idx):
        inputs, targets = batch[0].float().to(device), batch[1].float().to(device)

        with torch.no_grad():
            pred, _ = self(inputs[:, 0, :, :])

        # Compute the test loss, for example, using MSE Loss
        test_loss = F.mse_loss(pred, targets[:, 0, :, :, :])

        # self.log("test/loss", test_loss, prog_bar=True, sync_dist=self.sync_dist, logger=True)
        return test_loss
        
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer =  torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-4)
        
        return [optimizer]

    def save_checkpoint(self, filepath, weights_only:bool=False, storage_options:Optional[Any]=None) -> None:
        checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
        self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
        self.strategy.barrier("Trainer.save_checkpoint")

