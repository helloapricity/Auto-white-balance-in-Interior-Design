import argparse
import logging
import os
import sys
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from src import wb_net
import random
from src import ops
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

try:
  from torch.utils.tensorboard import SummaryWriter

  use_tb = True
except ImportError:
  use_tb = False

import wandb
import time

from src import dataset1
from datasets import load_dataset
from torch.utils.data import DataLoader

def ddp_setup(rank, world_size):
  """_summary_

  Args:
      rank (_type_): Unique indentifier of each process
      world_size (_type_): Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Define a helper function to log images
def log_images(tag, images, step, max_retries=5):
    for i, img in enumerate(images):
        retry_count = 0
        while retry_count < max_retries:
            try:
                wandb.log({f'{tag} {i + 1}': wandb.Image(img), 'global_step': step})
                break  # Exit loop if log is successful
            except wandb.errors.Error as e:
                if "rate limit exceeded" in str(e):
                    retry_count += 1
                    sleep_time = 2 ** retry_count  # Exponential backoff
                    print(f"Rate limit exceeded, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise e  # Reraise the exception if it's not rate limit related
                  
def train_net(rank, world_size, net, data_dir, val_dir=None, epochs=140,
              batch_size=32, lr=0.001, l2reg=0.00001, grad_clip_value=0,
              chkpoint_period=10, val_freq=1, smooth_weight=0.01,
              multiscale=False, wb_settings=None, shuffle_order=True,
              patch_number=12,  optimizer_algo='Adam', max_tr_files=0,
              max_val_files=0, patch_size=128, model_name='WB_model',
              save_cp=True):
  """ Trains a network and saves the trained model in harddisk.
  """
  ddp_setup(rank, world_size)
  torch.cuda.set_device(rank)
  device = torch.device(f'cuda:{rank}')
  # instantiate the model and move it to the right device
  net = net.to(device)
  # wrap the model with DDP
  net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=True)
  
  dir_checkpoint = 'checkpoints_model/'  # check points directory


  SMOOTHNESS_WEIGHT = smooth_weight


  train_data = load_dataset("Apricity0201/preprocess-presencesw-wb", split="train", streaming=True)
  validation_data = load_dataset("Apricity0201/preprocess-presencesw-wb", split="validation")
  
  train_set = dataset1.StreamingData(train_data, patch_size=4,
                           patch_number=4, multiscale=False,
                           shuffle_order=False, wb_settings=wb_settings)
  train_set = train_set.with_format("torch")
  
  val_set = dataset1.StreamingData(validation_data, patch_size=4,
                           patch_number=4, multiscale=False,
                           shuffle_order=False, wb_settings=wb_settings)
  val_set = iter(val_set)

  train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
  train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=6, pin_memory=True)

  val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank)
  val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=6, pin_memory=True)

  # Initialize W&B
  wandb.init(project="wb-correction", config={
      "model_name": model_name,
      "epochs": epochs,
      "batch_size": batch_size,
      "learning_rate": lr,
      "l2_reg_weight": l2reg,
      "validation_freq": val_freq,
      "grad_clipping": grad_clip_value,
      "patch_per_image": patch_number,
      "patch_size": patch_size,
      "optimizer": optimizer_algo,
      "smoothness_weight": smooth_weight,
      "shuffle_order": shuffle_order,
      "wb_settings": wb_settings,
      "model_name": model_name
  }, group="DDP")
  wandb.watch(net, log="all")

  if use_tb:  # if TensorBoard is used
    writer = SummaryWriter(log_dir='runs/' + model_name,
                           comment=f'LR_{lr}_BS_{batch_size}')
  else:
    writer = None
  global_step = 0

  logging.info(f'''Starting training:
        Model Name:            {model_name}
        Epochs:                {epochs}
        WB Settings:           {wb_settings}
        Batch size:            {batch_size}
        Patch per image:       {patch_number}
        Patch size:            {patch_size} x {patch_size}
        Learning rate:         {lr}
        L2 reg. weight:        {l2reg}
        Smooth weight:         {smooth_weight}
        Validation Freq.:      {val_freq}
        Grad. clipping:        {grad_clip_value}
        Optimizer:             {optimizer_algo}
        Checkpoints:           {save_cp}
        Device:                {device.type}
        TensorBoard:           {use_tb}
  ''')

  if optimizer_algo == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
                           weight_decay=l2reg)


  elif optimizer_algo == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=l2reg)

  else:
    raise NotImplementedError

  x_kernel, y_kernel = ops.get_sobel_kernel(device, chnls=len(wb_settings))

  for epoch in range(epochs):

    net.train()
    epoch_loss = 0
    epoch_smoothness_loss = 0
    epoch_rec_loss = 0


    with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1} / {epochs}',
              unit='img') as pbar:
      for batch in train_loader:
        img = batch['image']
        img = img.to(device=device, dtype=torch.float32)
        gt = batch['gt']
        gt = gt.to(device=device, dtype=torch.float32)
        rec_loss = 0
        smoothness_loss = 0

        for p in range(img.shape[1]):
          patch = img[:, p, :, :, :]
          gt_patch = gt[:, p, :, :, :]
          result, weights = net(patch)
          rec_loss += ops.compute_loss(result, gt_patch)

          smoothness_loss += SMOOTHNESS_WEIGHT * (
              torch.sum(F.conv2d(weights, x_kernel, stride=1) ** 2) +
              torch.sum(F.conv2d(weights, y_kernel, stride=1) ** 2))


        rec_loss = rec_loss / img.shape[1]
        smoothness_loss = smoothness_loss / img.shape[1]
        loss = rec_loss + smoothness_loss

        py_loss = loss.item()
        py_rec_loss = rec_loss.item()

        py_smoothness_loss = smoothness_loss.item()
        epoch_smoothness_loss += py_smoothness_loss

        epoch_rec_loss += py_rec_loss
        epoch_loss += py_loss



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if grad_clip_value > 0:
          torch.nn.utils.clip_grad_value_(net.parameters(), grad_clip_value)

        if use_tb:
          # for visualization
          vis_weights = (weights - torch.min(weights)) / (
              ops.EPS + torch.max(weights) - torch.min(weights))

          writer.add_scalar('Loss/train', py_loss, global_step)
          writer.add_scalar('Rec Loss/train', py_rec_loss, global_step)
          writer.add_scalar('Smoothness Loss/train', py_smoothness_loss,
                            global_step)

          writer.add_images('Input (1)', patch[:, 0:3, :, :], global_step)
          writer.add_images('Weight (1)',
                            torch.unsqueeze(vis_weights[:, 0, :, :], dim=1),
                            global_step)
          writer.add_images('Input (2)', patch[:, 3:6, :, :], global_step)
          writer.add_images('Weight (2)',
                            torch.unsqueeze(vis_weights[:, 1, :, :], dim=1),
                            global_step)
          writer.add_images('Input (3)', patch[:, 6:9, :, :], global_step)
          writer.add_images('Weight (3)',
                            torch.unsqueeze(vis_weights[:, 2, :, :], dim=1),
                            global_step)
          if vis_weights.shape[1] == 4:
            writer.add_images('Input (4)', patch[:, 9:12, :, :], global_step)
            writer.add_images('Weight (4)',
                              torch.unsqueeze(vis_weights[:, 3, :, :], dim=1),
                              global_step)
          if vis_weights.shape[1] == 5:
            writer.add_images('Input (4)', patch[:, 9:12, :, :], global_step)
            writer.add_images('Weight (4)',
                              torch.unsqueeze(vis_weights[:, 3, :, :], dim=1),
                              global_step)
            writer.add_images('Input (5)', patch[:, 12:, :, :], global_step)
            writer.add_images('Weight (5)',
                              torch.unsqueeze(vis_weights[:, 4, :, :], dim=1),
                              global_step)

          writer.add_images('Result', result, global_step)
          writer.add_images('GT', gt_patch, global_step)

        # Normalize weights for visualization
        vis_weights = (weights - torch.min(weights)) / (
                      torch.finfo(weights.dtype).eps + torch.max(weights) - torch.min(weights))

        # Log scalar metrics
        wandb.log({'Loss/train': py_loss, 'global_step': global_step})
        wandb.log({'Rec Loss/train': py_rec_loss, 'global_step': global_step})
        wandb.log({'Smoothness Loss/train': py_smoothness_loss, 'global_step': global_step})


        # Log input images and weights
        log_images('Input (1)', patch[:, 0:3, :, :], global_step)
        log_images('Weight (1)', torch.unsqueeze(vis_weights[:, 0, :, :], dim=1), global_step)
        log_images('Input (2)', patch[:, 3:6, :, :], global_step)
        log_images('Weight (2)', torch.unsqueeze(vis_weights[:, 1, :, :], dim=1), global_step)
        log_images('Input (3)', patch[:, 6:9, :, :], global_step)
        log_images('Weight (3)', torch.unsqueeze(vis_weights[:, 2, :, :], dim=1), global_step)

        if vis_weights.shape[1] == 4:
            log_images('Input (4)', patch[:, 9:12, :, :], global_step)
            log_images('Weight (4)', torch.unsqueeze(vis_weights[:, 3, :, :], dim=1), global_step)

        if vis_weights.shape[1] == 5:
            log_images('Input (4)', patch[:, 9:12, :, :], global_step)
            log_images('Weight (4)', torch.unsqueeze(vis_weights[:, 3, :, :], dim=1), global_step)
            log_images('Input (5)', patch[:, 12:, :, :], global_step)
            log_images('Weight (5)', torch.unsqueeze(vis_weights[:, 4, :, :], dim=1), global_step)

        # Log result and ground truth images
        log_images('Result', result, global_step)
        log_images('GT', gt_patch, global_step)
        
        pbar.update(np.ceil(img.shape[0]))

        pbar.set_postfix(**{'Total loss (batch)': py_loss},
                         **{'Rec. loss (batch)': py_rec_loss},
                         **{'Smoothness loss (batch)': py_smoothness_loss}
                         )

        global_step += 1

    # Calculating epoch losses
    epoch_loss = epoch_loss / (len(train_loader))
    epoch_rec_loss = epoch_rec_loss / (len(train_loader))
    epoch_smoothness_loss = epoch_smoothness_loss / (len(train_loader))
    logging.info(f'{model_name} - Epoch loss: = {epoch_loss}, '
                 f'Rec. loss = {epoch_rec_loss}, '
                 f'Smoothness loss = {epoch_smoothness_loss}')

    # Validation after each epoch
    if (epoch + 1) % val_freq == 0:
      logging.info('Validation...')
      validation(net=net, loader=val_loader, writer=writer, step=global_step, device=device)
    
    # Synchronize all processes
    dist.barrier()

    # save a checkpoint
    if rank == 0 and save_cp and (epoch + 1) % chkpoint_period == 0:
      if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
        logging.info('Created checkpoint directory')
      torch.save(net.module.state_dict(), dir_checkpoint +
                 f'{model_name}_{epoch + 1}.pth')
      logging.info(f'Checkpoint {epoch + 1} saved!')

  # save final trained model
  if not os.path.exists('models'):
    os.mkdir('models')
    logging.info('Created trained models directory')

  torch.save(net.module.state_dict(), 'models/' + f'{model_name}.pth')
  logging.info('Saved trained model!')

  if use_tb:
    writer.close()

  logging.info('End of training')
  
  # Finish W&B
  wandb.finish()
    
  destroy_process_group()

def validation(net, loader, writer, step, device):
  net.eval()
  index = random.randint(0, len(loader) - 1)
  val_loss = 0
  for b, batch in enumerate(loader):
    img = batch['image']
    img = img[:, 0, :, :, :]
    gt = batch['gt']
    gt = gt[:, 0, :, :, :]

    img = img.to(device=device, dtype=torch.float32)
    gt = gt.to(device=device, dtype=torch.float32)

    result, weights = net(img)

    val_loss = ops.compute_loss(result, gt)

    val_loss += val_loss.item()

    if b == index and writer is not None:
      # for visualization
      vis_weights = (weights - torch.min(weights)) / (
          ops.EPS + torch.max(weights) - torch.min(weights))
      writer.add_images('Input (1) [val]', img[:, 0:3, :, :], step)
      writer.add_images('Weight (1) [val]',
                        torch.unsqueeze(vis_weights[:, 0, :, :], dim=1),
                        step)
      writer.add_images('Input (2) [val]', img[:, 3:6, :, :], step)
      writer.add_images('Weight (2) [val]',
                        torch.unsqueeze(vis_weights[:, 1, :, :], dim=1),
                        step)
      writer.add_images('Input (3) [val]', img[:, 6:, :, :], step)
      writer.add_images('Weight (3) [val]',
                        torch.unsqueeze(vis_weights[:, 2, :, :], dim=1),
                        step)

      if vis_weights.shape[1] == 4:
        writer.add_images('Input (4) [val]', img[:, 9:12, :, :], step)
        writer.add_images('Weight (4) [val]',
                          torch.unsqueeze(vis_weights[:, 3, :, :], dim=1),
                          step)
      if vis_weights.shape[1] == 5:
        writer.add_images('Input (4) [val]', img[:, 9:12, :, :], step)
        writer.add_images('Weight (4) [val]',
                          torch.unsqueeze(vis_weights[:, 3, :, :], dim=1),
                          step)
        writer.add_images('Input (5) [val]', img[:, 12:, :, :], step)
        writer.add_images('Weight (5) [val]',
                          torch.unsqueeze(vis_weights[:, 4, :, :], dim=1),
                          step)

      writer.add_images('Result [val]', result, step)
      writer.add_images('GT [val]', gt, step)

    wandb.log({"validation_loss": val_loss / len(loader), "step": step})
    vis_weights = (weights - torch.min(weights)) / (ops.EPS + torch.max(weights) - torch.min(weights))
    # Log input images and weights
    log_images('Input (1)', img[:, 0:3, :, :], step)
    log_images('Weight (1)', vis_weights[:, 0, :, :].unsqueeze(1), step)
    log_images('Input (2)', img[:, 3:6, :, :], step)
    log_images('Weight (2)', vis_weights[:, 1, :, :].unsqueeze(1), step)
    log_images('Input (3)', img[:, 6:9, :, :], step)
    log_images('Weight (3)', vis_weights[:, 2, :, :].unsqueeze(1), step)

    if vis_weights.shape[1] == 4:
        log_images('Input (4)', img[:, 9:12, :, :], step)
        log_images('Weight (4)', vis_weights[:, 3, :, :].unsqueeze(1), step)

    if vis_weights.shape[1] == 5:
        log_images('Input (4)', img[:, 9:12, :, :], step)
        log_images('Weight (4)', vis_weights[:, 3, :, :].unsqueeze(1), step)
        log_images('Input (5)', img[:, 12:, :, :], step)
        log_images('Weight (5)', vis_weights[:, 4, :, :].unsqueeze(1), step)
    
    # Log result and ground truth images
    log_images('Result [val]', result, step)
    log_images('GT [val]', gt, step)

  print(f'Validation loss (batch): {val_loss / len(loader)}')
  if writer is not None:
    writer.add_scalar('Validation Loss', val_loss / len(loader), step)

  net.train()


def get_args():
  """ Gets command-line arguments.

  Returns:
    Return command-line arguments as a set of attributes.
  """

  parser = argparse.ArgumentParser(description='Train WB Correction.')
  parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                      help='Number of epochs', dest='epochs')

  parser.add_argument('-s', '--patch-size', dest='patch_size', type=int,
                      default=64, help='Size of input training patches')

  parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?',
                      default=8, help='Batch size', dest='batch_size')

  parser.add_argument('-pn', '--patch-number', type=int, default=4,
                      help='number of patches per trainig image',
                      dest='patch_number')

  parser.add_argument('-opt', '--optimizer', dest='optimizer', type=str,
                      default='Adam', help='Adam or SGD')

  parser.add_argument('-mtf', '--max-tr-files', dest='max_tr_files', type=int,
                      default=0, help='max number of training files; default '
                                      'is 0 which uses all files')

  parser.add_argument('-mvf', '--max-val-files', dest='max_val_files', type=int,
                      default=0, help='max number of validation files; '
                                       'default is 0 which uses all files')

  parser.add_argument('-nrm', '--normalization', dest='norm', type=bool,
                      default=False,
                      help='Apply BN in network')

  parser.add_argument('-msc', '--multi-scale', dest='multiscale', type=bool,
                      default=False,
                      help='Multi-scale training samples')

  parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float,
                      nargs='?', default=1e-4, help='Learning rate', dest='lr')

  parser.add_argument('-l2r', '--l2reg', metavar='L2Reg', type=float,
                      nargs='?', default=0, help='L2 regularization factor',
                      dest='l2r')

  parser.add_argument('-sw', '--smoothness-weight', dest='smoothness_weight',
                      type=float, default=100.0, help='smoothness weight')

  parser.add_argument('-wbs', '--wb-settings', dest='wb_settings', nargs='+',
                      default=['D', 'S', 'T', 'F', 'C'])

  parser.add_argument('-l', '--load', dest='load', type=bool, default=False,
                      help='Load model from a .pth file')

  parser.add_argument('-so', '--shuffle-order', dest='shuffle_order',
                      type=bool, default=False,
                      help='Shuffle order of WB')

  parser.add_argument('-ml', '--model-location', dest='model_location',
                      default=None)

  parser.add_argument('-vf', '--validation-frequency', dest='val_freq',
                      type=int, default=1, help='Validation frequency.')

  parser.add_argument('-cpf', '--checkpoint-frequency', dest='cp_freq',
                      type=int, default=1, help='Checkpoint frequency.')

  parser.add_argument('-gc', '--grad-clip-value', dest='grad_clip_value',
                      type=float, default=0, help='Gradient clipping value; '
                                                  'if = 0, no clipping applied')

  parser.add_argument('-trd', '--training-dir', dest='trdir',
                      default='./data/images/',
                      help='Training directory')

  parser.add_argument('-valdir', '--validation-dir', dest='valdir',
                      default=None, help='Main validation directory')

  parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int)

  parser.add_argument('-mn', '--model-name', dest='model_name', type=str,
                      default='WB_model', help='Model name')

  return parser.parse_args()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  logging.info('Training Mixed-Ill WB correction')
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device.type != 'cpu':
    torch.cuda.set_device(args.gpu)

  logging.info(f'Using device {device}')

  net = wb_net.WBnet(device=device, norm=args.norm, inchnls=3 * len(
    args.wb_settings))
  if args.load:
    net.load_state_dict(
      torch.load(args.model_location, map_location=device)
    )
    logging.info(f'Model loaded from {args.model_location}')

  postfix = f'_p_{args.patch_size}'

  if args.norm:
    postfix += f'_w_BN'

  if args.shuffle_order:
    postfix += f'_w_shuffling'

  if args.smoothness_weight == 0:
    postfix += f'_wo_smoothing'

  for wb_setting in args.wb_settings:
    postfix += f'_{wb_setting}'

  model_name = args.model_name + postfix

  world_size = torch.cuda.device_count()

  mp.spawn(train_net,
            args=(world_size, net, args.trdir, args.valdir, args.epochs, 
                  args.batch_size, args.lr, args.l2r, args.grad_clip_value, 
                  args.cp_freq, args.val_freq, args.smoothness_weight, 
                  args.multiscale, args.wb_settings, args.shuffle_order, 
                  args.patch_number, args.optimizer, args.max_tr_files,
                  args.max_val_files, args.patch_size, model_name, True),
            nprocs=world_size,
            join=True
            )