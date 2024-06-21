
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

    if self.mode == 'testing':
      self.deepWB_T = dwb.deepWBnet()
      self.deepWB_T.load_state_dict(torch.load('src/libs/models/net_t.pth'))
      self.deepWB_S = dwb.deepWBnet()
      self.deepWB_S.load_state_dict(torch.load('src/libs/models/net_s.pth'))
      self.deepWB_T.eval().to(device='cuda')
      self.deepWB_S.eval().to(device='cuda')

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
    
    base_name = ops.get_basename(inp_file[0]).split("/")[-1]
        
    if self.mode == 'testing':
      t_size = self.t_size
      full_size_img = img1.copy()

    if self.mode == 'training':
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

    else:  # testing mode

      checks = True

      if checks:
        if self.keep_aspect_ratio:
          img1 = ops.aspect_ratio_imresize(img1, max_output=t_size)
          img2 = ops.aspect_ratio_imresize(img2, max_output=t_size)
          img3 = ops.aspect_ratio_imresize(img3, max_output=t_size)
        else:
          img1 = ops.imresize.imresize(img1, output_shape=(t_size, t_size))
          img2 = ops.imresize.imresize(img2, output_shape=(t_size, t_size))
          img3 = ops.imresize.imresize(img3, output_shape=(t_size, t_size))

        mapping_2 = ops.get_mapping_func(img1, img2)
        full_size_2 = ops.apply_mapping_func(full_size_img, mapping_2)
        full_size_2 = ops.outOfGamutClipping(full_size_2)

        mapping_3 = ops.get_mapping_func(img1, img3)
        full_size_3 = ops.apply_mapping_func(full_size_img, mapping_3)
        full_size_3 = ops.outOfGamutClipping(full_size_3)

      else:
        img2, img3 = deep_wb(img1, task='editing', net_s=self.deepWB_S,
                            net_t=self.deepWB_T, device='cuda')
        if self.keep_aspect_ratio:
          img1 = ops.aspect_ratio_imresize(img1, max_output=t_size)
          img2 = ops.aspect_ratio_imresize(img2, max_output=t_size)
          img3 = ops.aspect_ratio_imresize(img3, max_output=t_size)
        else:
          img1 = ops.imresize.imresize(img1, output_shape=(t_size, t_size))
          img2 = ops.imresize.imresize(img2, output_shape=(t_size, t_size))
          img3 = ops.imresize.imresize(img3, output_shape=(t_size, t_size))

        mapping_2 = ops.get_mapping_func(img1, img2)
        mapping_3 = ops.get_mapping_func(img1, img3)
        full_size_2 = ops.apply_mapping_func(full_size_img, mapping_2)
        full_size_2 = ops.outOfGamutClipping(full_size_2)
        full_size_3 = ops.apply_mapping_func(full_size_img, mapping_3)
        full_size_3 = ops.outOfGamutClipping(full_size_3)

      img1 = ops.to_tensor(img1, dims=3)
      img2 = ops.to_tensor(img2, dims=3)
      img3 = ops.to_tensor(img3, dims=3)

      img = torch.cat((img1, img2, img3), dim=0)

      full_size_img = ops.to_tensor(full_size_img, dims=3)
      full_size_2 = ops.to_tensor(full_size_2, dims=3)
      full_size_3 = ops.to_tensor(full_size_3, dims=3)
      
      return {'image': img, 'fs_d_img': full_size_img, 'fs_s_img':
          full_size_2, 'fs_t_img': full_size_3, 'filename': base_name}
    
  @staticmethod
  def collate_fn(batch):
    inp_model, label = zip(*batch)
    
    inp_model =  torch.stack(inp_model)
    label = torch.stack(label)
    
    return inp_model, label

  @staticmethod
  def assert_files(files, wb_settings):
    for file in files:
      base_name = ops.get_basename(file)
      gt_img_file = os.path.basename(base_name) + 'G_AS.png'
      gt_img_file = os.path.join(os.path.split(os.path.dirname(file))[0],
                              'ground truth images', gt_img_file)
      s_img_file = base_name + 'S_CS.png'
      t_img_file = base_name + 'T_CS.png'
      paths = [file, gt_img_file, s_img_file, t_img_file]
      if 'F' in wb_settings:
        f_img_file = base_name + 'F_CS.png'
        paths.append(f_img_file)
      if 'C' in wb_settings:
        c_img_file = base_name + 'C_CS.png'
        paths.append(c_img_file)

      checks = True
      for curr_path in paths:
        checks = checks & os.path.exists(curr_path)
      assert checks, 'cannot find WB images match target WB settings'
    return True
  
  
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