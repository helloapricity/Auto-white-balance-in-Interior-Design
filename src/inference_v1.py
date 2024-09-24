import numpy as np
import glob
import torch
from utils import ops
from arguments import get_args
from models.wb_net import WBNet
from src.trainer_v2_AWB_StyleLoss_BatchNorm import LitAWB
from utils.ops import get_sobel_kernel
import os
import torch.nn.functional as F
import shutil
from tqdm import tqdm

class AWBInference():
    def __init__(self, net, t_size):
        self.t_size = t_size
        # self.data_dir = data_dir
        # self.out_dir = out_dir
        self.net = net

    def input_1_image(self, img_path):
        img = ops.imread(img_path)

        d_img = ops.to_tensor(img).unsqueeze(0).cuda(0)
        s_img = ops.to_tensor(img).unsqueeze(0).cuda(0)
        t_img = ops.to_tensor(img).unsqueeze(0).cuda(0)

        img = ops.imresize.imresize(img, output_shape=(self.t_size, self.t_size))

        batched_imgs = np.stack([img, img, img], axis=0).squeeze()
        # batched_imgs = ops.batch_aug(batched_imgs)

        inp_model = np.asarray(batched_imgs)

        inp_model = torch.as_tensor(inp_model.copy())
        num_inp, w, h, c = inp_model.shape
        inp_model = inp_model.reshape(num_inp * c, w, h)

        return inp_model, d_img, s_img, t_img

    def input_3_images(self, img1_path, img2_path, img3_path):
        img1 = ops.imread(img1_path)
        img2 = ops.imread(img2_path)
        img3 = ops.imread(img3_path)
        # print(img1.shape, img2.shape, img3.shape)
        d_img = ops.to_tensor(img1).unsqueeze(0).cuda(0)
        s_img = ops.to_tensor(img2).unsqueeze(0).cuda(0)
        t_img = ops.to_tensor(img3).unsqueeze(0).cuda(0)

        img1 = ops.imresize.imresize(img1, output_shape=(self.t_size, self.t_size))
        img2 = ops.imresize.imresize(img2, output_shape=(self.t_size, self.t_size))
        img3 = ops.imresize.imresize(img3, output_shape=(self.t_size, self.t_size))
        # print(img1.shape, img2.shape, img3.shape)
        
        batched_imgs = np.stack([img1, img2, img3], axis=0).squeeze()
        # print(batched_imgs.shape)

        inp_model = np.asarray(batched_imgs)
        inp_model = torch.as_tensor(inp_model.copy())
        num_inp, w, h, c = inp_model.shape
        inp_model = inp_model.reshape(num_inp * c, w, h)

        return inp_model, d_img, s_img, t_img
    
    def copy_subfolders(self, source_dir, destination_dir):
        # Ensure destination directory exists or create it
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # List all items in the source directory
        items = os.listdir(source_dir)

        # Iterate over items and copy subfolders
        for item in items:
            # Construct full path of the item
            item_path = os.path.join(source_dir, item)

            # Check if it's a directory (subfolder)
            if os.path.isdir(item_path):
                # Construct destination path
                destination_path = os.path.join(destination_dir, item)

                # Copy the subfolder recursively to the destination
                shutil.copytree(item_path, destination_path)
            
    def input_folder(self, data_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.copy_subfolders(data_dir, out_dir)
        
        img_folders = glob.glob(f'{data_dir}/*')
        for folder in tqdm(img_folders):
            img_files = glob.glob(os.path.join(folder, '*'))
            labels, inps = [], []
            for x in img_files:
                if '_G.jpg' in x:
                    labels.append(x)
                else:
                    inps.append(x)
            # print(inps)
            base_name = ops.get_basename(inps[0]).split("/")[-2]
            if len(img_files) >= 3:
                inp_model, d_img, s_img, t_img = self.input_3_images(inps[0], inps[1], inps[2])
            else:
                inp_model, d_img, s_img, t_img = self.input_1_image(img_files[0])

            with torch.no_grad():
                img = inp_model.to(device=device, dtype=torch.float32).unsqueeze(0)
                _, weights = self.net(img)
                print(f"Shape: {weights.shape}")

                # Ensure weights are interpolated to the correct size
                weights = F.interpolate(
                    weights, size=(d_img.shape[2], d_img.shape[3]), mode='bilinear', align_corners=True)

                imgs = [d_img, s_img, t_img]

                for i in range(weights.shape[1]):
                    if i == 0:
                        out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
                    else:
                        out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]

                # Print shapes for debugging
                # print(f'out_img shape: {out_img.shape}')
                # print(f'd_img shape: {d_img.shape}')

            result = ops.to_image(out_img[0, :, :, :])
            result.save(os.path.join(out_dir, base_name, base_name + "_output.png"))

def attem_load_author(model, checkpoint_path):
    # weights = torch.load(checkpoint_path)['state_dict']
    weights = torch.load(checkpoint_path)
    reweights = dict()
    for k, v in weights.items():
        reweights["model." + k] = v
        # reweights[k] = v
    
    model.load_state_dict(reweights)
    # model.eval()
    
    return model

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = WBNet(device=device, inchnls=3 * len(args.wb_settings))
    x_kernel, y_kernel = get_sobel_kernel(chnls=len(args.wb_settings))
    litmodel = LitAWB(model=net, lr=args.lr, smooth_weight=args.smoothness_weight, x_kernel=x_kernel, y_kernel=y_kernel)
    epoch = "06"
    model = f"sample-epoch={epoch}"
    model_path = f"output/{model}.ckpt"

    checkpoint = torch.load(model_path, map_location=device)

    # litmodel.load_state_dict(checkpoint["state_dict"])
    litmodel = attem_load_author(litmodel, model_path)
    litmodel.to(device=device)
    
    data_dir = "datahub/3_img/inference"
    out_dir = f"datahub/results/{model}_new_v7"
    t_size = 320
    shown = AWBInference(litmodel, t_size)
    shown.input_folder(data_dir, out_dir)