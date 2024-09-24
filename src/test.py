import argparse
import logging
import torch
from models.wb_net import WBNet
import os.path as path
import os
from utils import ops
from data.dataset import setup_dataset
import torch.nn.functional as F
from models import weight_refinement as weight_refinement
from arguments import get_args
from src.trainer_v1 import LitAWB
from utils.ops import get_sobel_kernel

def test_net(net, device, data_dir, model_name, out_dir, save_weights,
             multi_scale=False, keep_aspect_ratio=False, t_size=128,
             post_process=False, batch_size=32, wb_settings=None):
  """ Tests a trained network and saves the trained model in harddisk.
  """
  if wb_settings is None:
    wb_settings = ['D', 'S', 'T', 'F', 'C']
  
  test_dataloader = setup_dataset(
            imgfolders=args.testdir,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            patch_number=1,
            aug=False,
            mode='testing',
            multiscale=False,
            keep_aspect_ratio=False,
            t_size=args.img_size,
            num_workers=args.num_workers
        )


  logging.info(f'''Starting testing:
        Model Name:            {model_name}
        Batch size:            {batch_size}
        Output dir:            {out_dir}
        WB settings:           {wb_settings}
        Save weights:          {save_weights}
        Device:                {device.type}
  ''')

  if path.exists(out_dir) is not True:
    os.mkdir(out_dir)

  with torch.no_grad():

    for batch in test_dataloader:

      img = batch['image']

      img = img.to(device=device, dtype=torch.float32)
      _, weights = net(img)
      if multi_scale:
        img_1 = F.interpolate(
          img, size=(int(0.5 * img.shape[2]), int(0.5 * img.shape[3])),
          mode='bilinear', align_corners=True)
        _, weights_1 = net(img_1)
        weights_1 = F.interpolate(weights_1, size=(img.shape[2], img.shape[3]),
                                 mode='bilinear', align_corners=True)
        img_2 = F.interpolate(
          img, size=(int(0.25 * img.shape[2]), int(0.25 * img.shape[3])),
          mode='bilinear', align_corners=True)
        _, weights_2 = net(img_2)
        weights_2 = F.interpolate(weights_2, size=(img.shape[2], img.shape[3]),
                                 mode='bilinear', align_corners=True)
        weights = (weights + weights_1 + weights_2) / 3

      d_img = batch['fs_d_img']
      d_img = d_img.to(device=device, dtype=torch.float32)
      s_img = batch['fs_s_img']
      s_img = s_img.to(device=device, dtype=torch.float32)
      t_img = batch['fs_t_img']
      t_img = t_img.to(device=device, dtype=torch.float32)
      imgs = [d_img, s_img, t_img]
      if 'F' in wb_settings:
        f_img = batch['fs_f_img']
        f_img = f_img.to(device=device, dtype=torch.float32)
        imgs.append(f_img)
      if 'C' in wb_settings:
        c_img = batch['fs_c_img']
        c_img = c_img.to(device=device, dtype=torch.float32)
        imgs.append(c_img)

      filename = batch['filename']
      weights = F.interpolate(
        weights, size=(d_img.shape[2], d_img.shape[3]),
        mode='bilinear', align_corners=True)

      if post_process:
        for i in range(weights.shape[1]):
          for j in range(weights.shape[0]):
            ref = imgs[0][j, :, :, :]
            curr_weight = weights[j, i, :, :]
            refined_weight = weight_refinement.process_image(ref, curr_weight,
                                                             tensor=True)
            weights[j, i, :, :] = refined_weight
            weights = weights / torch.sum(weights, dim=1)


      for i in range(weights.shape[1]):
        if i == 0:
          out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
        else:
          out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]

      for i, fname in enumerate(filename):
        result = ops.to_image(out_img[i, :, :, :])
        name = path.join(out_dir, path.basename(fname) + '_WB.png')
        result.save(name)
        if save_weights:
          # save weights
          postfix = ['D', 'S', 'T']
          if 'F' in wb_settings:
            postfix.append('F')
          if 'C' in wb_settings:
            postfix.append('C')
          for j in range(weights.shape[1]):
            weight = torch.tile(weights[:, j, :, :], dims=(3, 1, 1))
            weight = ops.to_image(weight)
            name = path.join(out_dir, path.basename(fname) +
                             f'_weight_{postfix[j]}.png')
            weight.save(name)


  logging.info('End of testing')


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  logging.info('Testing Mixed-Ill WB correction')
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device.type != 'cpu':
    torch.cuda.set_device(args.gpu)

  logging.info(f'Using device {device}')

  net = WBNet(device=device, norm=args.norm, inchnls=3 * len(
    args.wb_settings))
  
  x_kernel, y_kernel = get_sobel_kernel(chnls=len(args.wb_settings))
  
  litmodel = LitAWB(model=net, lr=args.lr, smooth_weight=args.smoothness_weight,
                    x_kernel=x_kernel, y_kernel=y_kernel)

  model_path = os.path.join('checkpoints', args.model_name + '.ckpt')

  checkpoint = torch.load(model_path, map_location=device)
  
  litmodel.load_state_dict(checkpoint["state_dict"])

  logging.info(f'Model loaded from {model_path}')

  net.to(device=device)

  net.eval()

  test_net(net=net, device=device, data_dir=args.testdir,
           batch_size=args.batch_size, out_dir=args.outdir,
           post_process=args.post_process,
           keep_aspect_ratio=args.keep_aspect_ratio,
           t_size=args.img_size,
           multi_scale=args.multiscale, model_name=args.model_name,
           save_weights=args.save_weights,
           wb_settings=args.wb_settings)