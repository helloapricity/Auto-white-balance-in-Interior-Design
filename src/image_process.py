import torch 
import numpy as np 
from utils import ops
# def input_image(img_path):
#     img = ops.imread(img_path)
#     print(img.shape)
#     d_img = ops.to_tensor(img).unsqueeze(0).cuda(0)
#     s_img = ops.to_tensor(img).unsqueeze(0).cuda(0)
#     t_img = ops.to_tensor(img).unsqueeze(0).cuda(0)
#     img = ops.imresize.imresize(img, output_shape=(320, 320))

#     batched_imgs = np.stack([img, img, img], axis=0).squeeze()
#     inp_model = np.asarray(batched_imgs)
#     inp_model = torch.as_tensor(inp_model.copy())
#     print(inp_model.shape)
#     num_inp, w, h, c = inp_model.shape
#     inp_model = inp_model.reshape(num_inp * c, w, h)
#     print(inp_model.shape)
#     return inp_model, d_img, s_img, t_img

# img_path = "E:\\\\\\\\refactor\\\\\\\\refactor\\\\\\\\src\\\\\\\\datahub\\\\\\\\test_data\\\\\\\\30_1.png"
# inp_model, d_img, s_img, t_img = input_image(img_path)

# print("inp_model shape:", inp_model.shape)
# print("d_img shape:", d_img.shape)
# print("s_img shape:", s_img.shape)
# print("t_img shape:", t_img.shape)

def input_3_images(img1_path, img2_path, img3_path):
        img1 = ops.imread(img1_path)
        img2 = ops.imread(img2_path)
        img3 = ops.imread(img3_path)

        d_img = ops.to_tensor(img1).unsqueeze(0).cuda(0)
        s_img = ops.to_tensor(img2).unsqueeze(0).cuda(0)
        t_img = ops.to_tensor(img3).unsqueeze(0).cuda(0)

        img1 = ops.imresize.imresize(img1, output_shape=(320, 320))
        img2 = ops.imresize.imresize(img2, output_shape=(320, 320))
        img3 = ops.imresize.imresize(img3, output_shape=(320, 320))
        
        batched_imgs = np.stack([img1, img2, img3], axis=0).squeeze()
        inp_model = np.asarray(batched_imgs)
        inp_model = torch.as_tensor(inp_model.copy())
        num_inp, w, h, c = inp_model.shape
        inp_model = inp_model.permute(0, 3, 1, 2)
        inp_model = inp_model.reshape(num_inp * c, w, h)

        return inp_model, d_img, s_img, t_img

img1_path = "E:\\refactor\\refactor\\src\\datahub\\test_data\\1\\1_1.png"
img2_path = "E:\\refactor\\refactor\\src\\datahub\\test_data\\1\\1_2.png"
img3_path = "E:\\refactor\\refactor\\src\\datahub\\test_data\\1\\1_3.png"
inp_model, d_img, s_img, t_img = input_3_images(img1_path, img2_path, img3_path)
print("inp shape", inp_model.shape)