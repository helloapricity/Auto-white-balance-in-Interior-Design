
import os
import glob
import torch
import logging
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils import ops
from libs.arch import deep_wb_single_task as dwb
from libs.utilities.deepWB import deep_wb



class AWBData(Dataset):
  def __init__(self, imgfolders, patch_size=128, patch_number=32, 
              aug=True, mode='training',
              multiscale=False, keep_aspect_ratio=False, t_size=320):
    """ Data constructor
    
    """

    self.imgfolders = glob.glob(f'{imgfolders}/*')
    self.patch_size = patch_size
    self.patch_number = patch_number
    self.keep_aspect_ratio = keep_aspect_ratio
    self.aug = aug
    self.multiscale = multiscale
    self.mode = mode

    self.t_size = t_size

    logging.info(f'Creating dataset with {len(self.imgfolders)} examples')

  def __len__(self):
    """ Gets length of image files in the dataloader. """

    return len(self.imgfolders)

  def __getitem__(self, i):
    """ Gets next data in the dataloader.

    Args:
      i: index of file in the dataloader.

    Returns:
      A dictionary of the following keys:
      - image:
    """

    img_folder = self.imgfolders[i]    
    samples = glob.glob(f'{img_folder}/*')

    labels, inps = [], []
    for x in samples:
      if '_G.png' in x:
        labels.append(x)
      else:
        inps.append(x)

    if len(inps) > 3:
      random.Random(42).shuffle(inps)
      inp_file = inps[:3]
    else:
      inp_file = inps

    label = ops.imread(labels[0])
    img1 = ops.imread(inp_file[0])
    img2 = ops.imread(inp_file[1])
    img3 = ops.imread(inp_file[2])
    
    t_size = self.t_size + 64 * 2 ** np.random.randint(5) if self.multiscale else self.t_size
    
    gt_img = ops.imresize.imresize(label, output_shape=(t_size, t_size))
    img2 = ops.imresize.imresize(img2, output_shape=(t_size, t_size))
    img1 = ops.imresize.imresize(img1, output_shape=(t_size, t_size))
    img3 = ops.imresize.imresize(img3, output_shape=(t_size, t_size))
    
    # Ground_truth at the first position 
    batched_imgs = np.stack([gt_img, img1, img2, img3], axis=0).squeeze()
    
    if self.aug:
        batched_imgs = ops.batch_aug(batched_imgs)
    
    batched_imgs = ops.batch_extract_path(batched_imgs, patch_size=self.patch_size, patch_number=self.patch_number)

    # Convert      
    label = np.asarray(batched_imgs[:,0,:,:])
    inp_model = np.asarray(batched_imgs[:,1:,:,:])
    
    label = torch.as_tensor(label).permute(0, 3, 1, 2)
    inp_model = torch.as_tensor(inp_model)
    
    del batched_imgs
    
    patch, num_inp, w, h, c = inp_model.shape
    inp_model = inp_model.reshape(patch, num_inp*c, w, h)
    
    return inp_model, label
    
  @staticmethod
  def collate_fn(batch):
    inp_model, label = zip(*batch)
    
    inp_model =  torch.stack(inp_model)
    label = torch.stack(label)
    
    return inp_model, label 
  
def setup_dataset(imgfolders, batch_size, patch_size, patch_number, aug, 
                  mode, multiscale, keep_aspect_ratio, t_size, num_workers):
  
  training_data = AWBData(imgfolders, patch_size, patch_number,
                          aug, mode, multiscale, keep_aspect_ratio,
                          t_size)
  
  traininng_loader = DataLoader(
    training_data, 
    batch_size=batch_size if mode == 'training' else batch_size*2,
    shuffle=True if mode == 'training' else False,
    num_workers=num_workers,
    collate_fn=AWBData.collate_fn
  )
  
  return traininng_loader