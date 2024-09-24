import numpy as np
import glob
import torch
from utils import ops
from arguments import get_args
from models.wb_net import WBNet
from trainer_v3_StyleLoss_LCNorm import LitAWB
from utils.ops import get_sobel_kernel
import os
import torch.nn.functional as F
import shutil
from tqdm import tqdm
from models.style_model import StyleModel
from models import weight_refinement as weight_refinement

class AWBInference():
    def __init__(self, net, t_size, post_process=True):
        self.t_size = t_size
        self.net = net
        self.post_process = post_process

    def input_1_image(self, img_path):
        img = ops.imread(img_path)
        d_img = ops.to_tensor(img).unsqueeze(0).cuda(0)
        s_img = ops.to_tensor(img).unsqueeze(0).cuda(0)
        t_img = ops.to_tensor(img).unsqueeze(0).cuda(0)
        img = ops.imresize.imresize(img, output_shape=(self.t_size, self.t_size))

        batched_imgs = np.stack([img, img, img], axis=0).squeeze()
        inp_model = np.asarray(batched_imgs)
        inp_model = torch.as_tensor(inp_model.copy())
        num_inp, w, h, c = inp_model.shape
        inp_model = inp_model.reshape(num_inp * c, w, h)

        return inp_model, d_img, s_img, t_img

    def input_3_images(self, img1_path, img2_path, img3_path):
        img1 = ops.imread(img1_path)
        img2 = ops.imread(img2_path)
        img3 = ops.imread(img3_path)

        d_img = ops.to_tensor(img1).unsqueeze(0).cuda(0)
        s_img = ops.to_tensor(img2).unsqueeze(0).cuda(0)
        t_img = ops.to_tensor(img3).unsqueeze(0).cuda(0)

        img1 = ops.imresize.imresize(img1, output_shape=(self.t_size, self.t_size))
        img2 = ops.imresize.imresize(img2, output_shape=(self.t_size, self.t_size))
        img3 = ops.imresize.imresize(img3, output_shape=(self.t_size, self.t_size))
        
        batched_imgs = np.stack([img1, img2, img3], axis=0).squeeze()
        inp_model = np.asarray(batched_imgs)
        inp_model = torch.as_tensor(inp_model.copy())
        num_inp, w, h, c = inp_model.shape
        inp_model = inp_model.reshape(num_inp * c, w, h)

        return inp_model, d_img, s_img, t_img
    
    def copy_subfolders(self, source_dir, destination_dir):
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        items = os.listdir(source_dir)
        for item in items:
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path):
                destination_path = os.path.join(destination_dir, item)
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
            base_name = ops.get_basename(inps[0]).split("/")[-2]
            if len(img_files) >= 3:
                inp_model, d_img, s_img, t_img = self.input_3_images(inps[0], inps[1], inps[2])
            else:
                inp_model, d_img, s_img, t_img = self.input_1_image(img_files[0])

            with torch.no_grad():
                img = inp_model.to(device=device, dtype=torch.float32).unsqueeze(0)
                _, weights = self.net(img)

                weights = F.interpolate(weights, size=(d_img.shape[2], d_img.shape[3]), mode='bilinear', align_corners=True)
                imgs = [d_img, s_img, t_img]

                if self.post_process:
                    for i in range(weights.shape[1]):
                        for j in range(weights.shape[0]):
                            ref = imgs[0][j, :, :, :]
                            curr_weight = weights[j, i, :, :]
                            refined_weight = weight_refinement.process_image(ref, curr_weight, tensor=True)
                            weights[j, i, :, :] = refined_weight
                    weights = weights / torch.sum(weights, dim=1)

                for i in range(weights.shape[1]):
                    if i == 0:
                        out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
                    else:
                        out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]

            result = ops.to_image(out_img[0, :, :, :])
            result.save(os.path.join(out_dir, base_name, base_name + "_output.png"))

def attem_load_author(model, checkpoint_path):
    # weights = torch.load(checkpoint_path)['state_dict']
    weights = torch.load(checkpoint_path)
    # print(weights['state_dict'].keys())
    # print('-----------------')
    # print(model)
    reweights = dict()
    for k, v in weights['state_dict'].items():
        # print(k)
        # flag = 1
        # for key in ["model.net.epoch", "model.net.global_step", "model.net.pytorch-lightning_version", "model.net.state_dict", "model.net.loops"]:
        #     if key in k:
        #         flag = 0
        # if flag == 0:
        #     continue       
        # # reweights["model." + k] = v
        # if 'style' in k:
        #     continue
        reweights[k] = v

# def attem_load_author(model, checkpoint_path):
#     weights = torch.load(checkpoint_path)
#     # weights = torch.load(checkpoint_path)
#     reweights = dict()
#     for k, v in weights.items():
#         # print(k)
#         reweights["model."+ k] = v
        
#         # print(k)
#         # reweights[k[6:]] = v
        
    
    # model.load_state_dict(reweights, strict=False)
    model.load_state_dict(reweights)
    model.eval()
    
    return model

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg16 = StyleModel()
    net = WBNet(device=device, norm=args.norm, inchnls=3 * len(args.wb_settings))
    x_kernel, y_kernel = get_sobel_kernel(chnls=len(args.wb_settings))
    litmodel = LitAWB(model=net, lr=args.lr, smooth_weight=args.smoothness_weight, x_kernel=x_kernel, y_kernel=y_kernel, style_model = vgg16)
    epoch = "98-v12"
    model = f"sample-epoch={epoch}"
    model_path = f"output-v4/{model}.ckpt"
    # model_path = f"/home/tiennv/FPT/training_wb/weights/WB_model_p_64_D_S_T.pth"

    checkpoint = torch.load(model_path, map_location=device)

    litmodel = attem_load_author(litmodel, model_path)
    litmodel.to(device=device)
    
    data_dir = "datahub/3_img/inference"
    out_dir = f"datahub/results/version_12/{model}"
    t_size = 320
    shown = AWBInference(litmodel, t_size, post_process=True)
    shown.input_folder(data_dir, out_dir)