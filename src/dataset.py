from os.path import join
from os import listdir
from os import path
import random
import glob
from torch.utils.data import Dataset
import logging
from src import ops
import torch
import numpy as np
from DeepWB.arch import deep_wb_single_task as dwb
from DeepWB.utilities.deepWB import deep_wb
from DeepWB.utilities.utils import colorTempInterpolate_w_target
import time



class Data(Dataset):
  def __init__(self, imgfolders, patch_size=128, patch_number=32, aug=True,
               shuffle_order=False, mode='training',
               multiscale=False, keep_aspect_ratio=False, t_size=320):
    """ Data constructor
    """

    self.imgfolders = imgfolders
    self.patch_size = patch_size
    self.patch_number = patch_number
    self.keep_aspect_ratio = keep_aspect_ratio
    self.aug = aug
    self.multiscale = multiscale
    self.shuffle_order = shuffle_order
    assert (mode == 'training' or
            mode == 'testing'), 'mode should be training or testing'
    self.mode = mode

    if shuffle_order is True and self.mode == 'testing':
      logging.warning('Shuffling is not allowed in testing mode')
      self.shuffle_order = False

    self.t_size = t_size

    if self.mode == 'testing':
      self.deepWB_T = dwb.deepWBnet()
      self.deepWB_T.load_state_dict(torch.load('DeepWB/models/net_t.pth'))
      self.deepWB_S = dwb.deepWBnet()
      self.deepWB_S.load_state_dict(torch.load('DeepWB/models/net_s.pth'))
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

    # if len(inps) < 3:
    #   inp_file = random.choices(inps, k=3)
    if len(inps) > 3:
      random.Random(42).shuffle(inps)
      inp_file = inps[:3]
    else:
      inp_file = inps

    label = ops.imread(labels[0])
    img1 = ops.imread(inp_file[0])
    img2 = ops.imread(inp_file[1])
    img3 = ops.imread(inp_file[2])
    
    if self.mode == 'testing':
      t_size = self.t_size
      full_size_img = img_1.copy()

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
      
      batched_imgs = ops.batch_extract_path(batched_imgs, patch_size=self.patch_size, patch_number=self.patch_number)  # Total time extract modify:  0.1757 s
      
      # Total time convert modify: 0.0006
      label = torch.from_numpy(batched_imgs[:,0,:,:].copy()).permute(0, 3, 1, 2)
      inp_model = torch.from_numpy(batched_imgs[:,1:,:,:].copy())
      patch, num_inp, w, h, c = inp_model.shape
      inp_model = inp_model.reshape(patch, num_inp*c, w, h)
      
      return {'image':inp_model, 'gt':label}

    else:  # testing mode

      checks = True

      if checks:
        if self.keep_aspect_ratio:
          img_1 = ops.aspect_ratio_imresize(img_1, max_output=t_size)
        else:
          img_1 = ops.imresize.imresize(img_1, output_shape=(t_size, t_size))
          
        if self.keep_aspect_ratio:
          img_2 = ops.aspect_ratio_imresize(img_2, max_output=t_size)
        else:
          img_2 = ops.imresize.imresize(img_2, output_shape=(t_size, t_size))
        mapping_2 = ops.get_mapping_func(img_1, img_2)
        full_size_2 = ops.apply_mapping_func(full_size_img, mapping_2)
        full_size_2 = ops.outOfGamutClipping(full_size_2)

        if self.keep_aspect_ratio:
          img_3 = ops.aspect_ratio_imresize(img_3, max_output=t_size)
        else:
          img_3 = ops.imresize.imresize(img_3, output_shape=(t_size, t_size))
        mapping_3 = ops.get_mapping_func(img_1, img_3)
        full_size_3 = ops.apply_mapping_func(full_size_img, mapping_3)
        full_size_3 = ops.outOfGamutClipping(full_size_3)

      else:
        img_2, img_3 = deep_wb(img_1, task='editing', net_s=self.deepWB_S,
                               net_t=self.deepWB_T, device='cuda')
        if self.keep_aspect_ratio:
          img_1 = ops.aspect_ratio_imresize(img_1, max_output=t_size)
          img_2 = ops.aspect_ratio_imresize(img_2, max_output=t_size)
          img_3 = ops.aspect_ratio_imresize(img_3, max_output=t_size)
        else:
          img_1 = ops.imresize.imresize(img_1, output_shape=(t_size, t_size))
          img_2 = ops.imresize.imresize(img_2, output_shape=(t_size, t_size))
          img_3 = ops.imresize.imresize(img_3, output_shape=(t_size, t_size))

        mapping_2 = ops.get_mapping_func(img_1, img_2)
        mapping_3 = ops.get_mapping_func(img_1, img_3)
        full_size_2 = ops.apply_mapping_func(full_size_img, mapping_2)
        full_size_2 = ops.outOfGamutClipping(full_size_2)
        full_size_3 = ops.apply_mapping_func(full_size_img, mapping_3)
        full_size_3 = ops.outOfGamutClipping(full_size_3)

      img_1 = ops.to_tensor(img_1, dims=3)
      img_2 = ops.to_tensor(img_2, dims=3)
      img_3 = ops.to_tensor(img_3, dims=3)

      img = torch.cat((img_1, img_2, img_3), dim=0)

      full_size_img = ops.to_tensor(full_size_img, dims=3)
      full_size_2 = ops.to_tensor(full_size_2, dims=3)
      full_size_3 = ops.to_tensor(full_size_3, dims=3)
      
      return {'image': img, 'fs_d_img': full_size_img, 'fs_s_img':
          full_size_2, 'fs_t_img': full_size_3}

  def check_img_folder(self, folder_path):
    
    has_g_img = False
    count_other = 0

    for file_name in listdir(folder_path):
        if file_name.endswith('_G.png'):
            has_g_img = True
        else:
            count_other += 1
    return has_g_img, count_other

  @staticmethod
  def load_folders(folder_dir, mode='training'):
    """ Loads foldernames in a given data directory.

    Args:
      folder_dir: folder directory.

    Returns:
      imgfolders: a list of full foldernames.
    """
    logging.info(f'Loading folders information from {folder_dir}...')
    imgfolders = [join(folder_dir, folder) for folder in listdir(folder_dir)]
    
    return imgfolders

  @staticmethod
  def assert_files(files, wb_settings):
    for file in files:
      base_name = ops.get_basename(file)
      gt_img_file = path.basename(base_name) + 'G_AS.png'
      gt_img_file = path.join(path.split(path.dirname(file))[0],
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
        checks = checks & path.exists(curr_path)
      assert checks, 'cannot find WB images match target WB settings'
    return True
