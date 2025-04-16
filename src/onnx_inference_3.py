import numpy as np
import glob
import torch
from utils import ops
from arguments import get_args
from models.litawb_style_module import LitAWBStyleLoss
from utils.ops import get_sobel_kernel
import os
from models.wb_net import WBNet
from models.vgg19 import vgg19_net
import torch.nn.functional as F
import shutil
from tqdm import tqdm
import models.weight_refinement as weight_refinement
import onnx
import onnxruntime as ort


class AWBInference():
    def __init__(self, net, t_size, post_process=True, onnx_model_path=None):
        self.t_size = t_size
        self.net = net
        self.post_process = post_process
        self.onnx_model_path = onnx_model_path
        
        # Nếu có đường dẫn đến model ONNX, khởi tạo phiên làm việc ONNX
        if onnx_model_path:
            self.onnx_session = ort.InferenceSession(onnx_model_path)
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            self.onnx_output_name = self.onnx_session.get_outputs()[0].name

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
        inp_model = inp_model.permute(0, 3, 1, 2)
        inp_model = inp_model.reshape(num_inp * c, w, h)

        return inp_model, d_img, s_img, t_img
    
    # def copy_subfolders(self, source_dir, destination_dir):
    #     if not os.path.exists(destination_dir):
    #         os.makedirs(destination_dir)
    #     items = os.listdir(source_dir)
    #     for item in items:
    #         item_path = os.path.join(source_dir, item)
    #         if os.path.isdir(item_path):
    #             destination_path = os.path.join(destination_dir, item)
    #             shutil.copytree(item_path, destination_path, dirs_exist_ok=True)
    def rename_images(self, folder):
        """
        Đổi tên ảnh trong thư mục thành định dạng: <folder_name>_1.png, <folder_name>_2.png, <folder_name>_3.png.

        Args:
            folder (str): Đường dẫn đến thư mục chứa ảnh.

        Returns:
            list: Danh sách đường dẫn của các file đã đổi tên.
        """
        img_files = sorted(glob.glob(os.path.join(folder, "*")))  # Sắp xếp theo thứ tự
        folder_name = os.path.basename(folder)
        renamed_paths = []

        for i, img_path in enumerate(img_files[:3], start=1):  # Giới hạn chỉ lấy 3 ảnh đầu tiên
            new_name = f"{folder_name}_{i}.png"
            new_path = os.path.join(folder, new_name)
            os.rename(img_path, new_path)  # Đổi tên file
            renamed_paths.append(new_path)

        return renamed_paths

    def input_folder(self, data_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        # self.copy_subfolders(data_dir, out_dir)
        
        img_folders = glob.glob(f'{data_dir}/*')
        for folder in tqdm(img_folders):
            img_files = glob.glob(os.path.join(folder, '*'))
            labels, inps = [], []
            number = os.path.basename(folder)
            inps = self.rename_images(folder)
            labels = [os.path.join(folder, f"{number}_G.jpg")]
            
            base_name = os.path.basename(os.path.dirname(inps[0]))
            print(f"base_name: {base_name}")
            
            if len(img_files) >= 3:
                inp_model, d_img, s_img, t_img = self.input_3_images(inps[0], inps[1], inps[2])
            else:
                inp_model, d_img, s_img, t_img = self.input_1_image(img_files[0])

            # Chạy ONNX inference nếu mô hình ONNX được cung cấp
            if self.onnx_model_path:
                raw_outputs = self.onnx_inference(inp_model)
                weights = raw_outputs[0]  # Giả sử đầu ra là weights
                print("Weights shape from ONNX:", weights.shape)
            else:
                # Nếu không sử dụng ONNX, chạy mô hình PyTorch như bình thường
                with torch.no_grad():
                    img = inp_model.to(device=device, dtype=torch.float32).unsqueeze(0)
                    _, weights = self.net(img)
                print("Weights shape from PyTorch:", weights.shape)

            # Xử lý weights
            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights).float()
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
                    weights = weights.to(device)
                    out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i].to(device)
                else:
                    out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]

            result = ops.to_image(out_img[0, :, :, :])
            result.save(os.path.join(out_dir, base_name, base_name + "_output.png"))

    def onnx_inference(self, inp_model):
        """
        Hàm inference với mô hình ONNX
        """
        # Đảm bảo inp_model có kiểu dữ liệu là float32 và shape phù hợp (1, 9, 320, 320)
        if isinstance(inp_model, torch.Tensor):
            inp_model = inp_model.to(dtype=torch.float32)  # Đảm bảo kiểu dữ liệu là float32

        # Đảm bảo inp_model có shape (1, 9, t_size, t_size) nếu model của bạn yêu cầu
        inp_model = inp_model.unsqueeze(0)  # Thêm batch dimension nếu cần, shape trở thành (1, 9, t_size, t_size)

        # Đảm bảo inp_model là tensor với shape (1, 9, t_size, t_size)
        raw_outputs = self.onnx_session.run([self.onnx_output_name], {self.onnx_input_name: inp_model.numpy()})

        return raw_outputs
    def gradio_process(self, img1, img2, img3):
        """
        Hàm xử lý Gradio: Nhận 3 ảnh đầu vào, chạy qua pipeline và trả về ảnh output.
        Args:
            img1, img2, img3: Các ảnh đầu vào từ Gradio (PIL Images).
        Returns:
            PIL.Image: Ảnh kết quả.
        """
        # Lưu ảnh đầu vào tạm thời
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        temp_folder = "temp"
        os.makedirs(temp_folder, exist_ok=True)

        img1_path = os.path.join(temp_folder, "input_1.png")
        img2_path = os.path.join(temp_folder, "input_2.png")
        img3_path = os.path.join(temp_folder, "input_3.png")

        img1.save(img1_path)
        img2.save(img2_path)
        img3.save(img3_path)

        # Xử lý qua pipeline (chạy inference với input_3_images)
        inp_model, d_img, s_img, t_img = self.input_3_images(img1_path, img2_path, img3_path)

        # Chạy ONNX inference
        if self.onnx_model_path:
            raw_outputs = self.onnx_inference(inp_model)
            weights = raw_outputs[0]
        else:
            with torch.no_grad():
                img = inp_model.to(device=device, dtype=torch.float32).unsqueeze(0)
                _, weights = self.net(img)

        # Xử lý hậu kỳ weights (post-process)
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float()
        weights = F.interpolate(weights, size=(d_img.shape[2], d_img.shape[3]), mode="bilinear", align_corners=True)

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
                weights = weights.to(device)
                out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i].to(device)
            else:
                out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]

        # Chuyển đổi kết quả sang định dạng PIL Image để trả về
        result = ops.to_image(out_img[0, :, :, :])
        return result



def attem_load_author(model, checkpoint_path):
    weights = torch.load(checkpoint_path)
    reweights = dict()
    for k, v in weights['state_dict'].items():
        reweights[k] = v

    model.load_state_dict(reweights)
    model.eval()
    return model


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = WBNet(norm=args.norm, inchnls=3 * len(args.wb_settings))
    x_kernel, y_kernel = get_sobel_kernel(chnls=len(args.wb_settings))
    litmodel = LitAWBStyleLoss(model=net, lr=args.lr, smooth_weight=args.smoothness_weight, x_kernel=x_kernel, y_kernel=y_kernel, vgg_model=vgg19_net)
    
    epoch = "141"
    model = f"sample-epoch={epoch}"
    model_path = os.path.join(os.path.dirname(__file__), "..", "output", f"{model}.ckpt")
    checkpoint = torch.load(model_path, map_location=device)

    litmodel = attem_load_author(litmodel, model_path)
    litmodel.to(device=device)
    
    # Dưới đây là ví dụ sử dụng mô hình ONNX thay vì PyTorch
    onnx_model_path = "wbnet_model.onnx"
    
    t_size = 320
    shown = AWBInference(net = None, t_size=t_size, post_process=True, onnx_model_path=onnx_model_path)
    data_dir = os.path.join("datahub", "test_data")
    out_dir = os.path.join("datahub", "results", "AWB_style_loss_author_dataset", model)
    shown.input_folder(data_dir, out_dir)
