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
import cv2

    
class AWBInference():
    def __init__(self, net, t_size, post_process=True, onnx_model_path=None):
        self.t_size = t_size
        self.net = net
        self.post_process = post_process
        self.onnx_model_path = onnx_model_path

        # Khởi tạo ONNX runtime nếu có
        if onnx_model_path:
            self.onnx_session = ort.InferenceSession(onnx_model_path)
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            self.onnx_output_name = self.onnx_session.get_outputs()[0].name

    def apply_exposure_adjustment(self, img, temp_dir):
        """Tạo ảnh điều chỉnh phơi sáng và lưu vào `temp_dir`."""
        gamma_values = [2.5, 4.5]
        linear_factors = [2.0, 4.0]
        filenames = []
        os.makedirs(temp_dir, exist_ok=True)

        # Gamma correction
        for idx, gamma in enumerate(gamma_values, start=1):
            gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
            adjusted_img = cv2.LUT(img, gamma_table)
            filename = os.path.join(temp_dir, f"gamma_{idx}.png")
            cv2.imwrite(filename, adjusted_img)
            filenames.append(filename)

        # Original image
        original_path = os.path.join(temp_dir, "original.png")
        cv2.imwrite(original_path, img)
        filenames.append(original_path)

        # Linear scaling
        for idx, factor in enumerate(linear_factors, start=1):
            adjusted_img = np.clip(img * factor, 0, 255).astype(np.uint8)
            filename = os.path.join(temp_dir, f"linear_{idx}.png")
            cv2.imwrite(filename, adjusted_img)
            filenames.append(filename)

        return filenames

    def create_hdr_image(self, image_paths, output_path):
        """Tạo ảnh HDR từ các ảnh điều chỉnh phơi sáng."""
        exposure_times = np.array([15.0, 2.5, 0.25, 0.0333, 0.01], dtype=np.float32)
        img_list = [cv2.imread(path) for path in image_paths]

        if any(img is None for img in img_list):
            raise ValueError("Một hoặc nhiều ảnh không thể được tải lên!")

        # Tạo HDR với Mertens
        merge_mertens = cv2.createMergeMertens()
        hdr_image = merge_mertens.process(img_list)
        hdr_8bit = np.clip(hdr_image * 255, 0, 255).astype("uint8")

        # Lưu HDR
        cv2.imwrite(output_path, hdr_8bit)

    def adjust_exposure(self, img, output_paths):
        """Tạo 2 ảnh: sáng hơn và tối hơn từ HDR."""
        gamma_values = [0.5, 1.5]  # Sáng hơn và tối hơn
        for gamma, output_path in zip(gamma_values, output_paths):
            gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
            adjusted_img = cv2.LUT(img, gamma_table)
            cv2.imwrite(output_path, adjusted_img)

    # def adjust_exposure(self, img, output_paths, gamma=2.5, linear_factor=2.0):
    #     """
    #     Tạo 2 ảnh: ảnh tối hơn sử dụng gamma và ảnh sáng hơn sử dụng linear scaling.
    #     Args:
    #         img: Ảnh gốc (numpy array).
    #         output_paths: Danh sách đường dẫn để lưu 2 ảnh đầu ra.
    #         gamma: Hệ số gamma để tạo ảnh tối.
    #         linear_factor: Hệ số linear để tạo ảnh sáng hơn.
    #     """
    #     # Kiểm tra output_paths có đủ 2 đường dẫn
    #     if len(output_paths) != 2:
    #         raise ValueError("output_paths phải chứa chính xác 2 đường dẫn cho ảnh tối và sáng.")

    #     # Ảnh tối bằng gamma correction
    #     gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    #     dark_img = cv2.LUT(img, gamma_table)
    #     cv2.imwrite(output_paths[0], dark_img)

    #     # Ảnh sáng bằng linear scaling
    #     bright_img = np.clip(img * linear_factor, 0, 255).astype(np.uint8)
    #     cv2.imwrite(output_paths[1], bright_img)

        

    def input_3_images(self, img1_path, img2_path, img3_path):
            """
            Chuẩn bị 3 ảnh đầu vào để tạo tensor cho mô hình.
            """
            # Đọc ảnh đầu vào
            img1 = ops.imread(img1_path)
            img2 = ops.imread(img2_path)
            img3 = ops.imread(img3_path)

            # Chuyển ảnh sang tensor và thêm batch dimension
            d_img = ops.to_tensor(img1).unsqueeze(0).cuda(0)
            s_img = ops.to_tensor(img2).unsqueeze(0).cuda(0)
            t_img = ops.to_tensor(img3).unsqueeze(0).cuda(0)

            # Resize ảnh về kích thước chuẩn
            img1 = ops.imresize.imresize(img1, output_shape=(self.t_size, self.t_size))
            img2 = ops.imresize.imresize(img2, output_shape=(self.t_size, self.t_size))
            img3 = ops.imresize.imresize(img3, output_shape=(self.t_size, self.t_size))

            # Kết hợp ảnh thành batch và chuyển sang tensor
            batched_imgs = np.stack([img1, img2, img3], axis=0).squeeze()
            inp_model = np.asarray(batched_imgs)
            inp_model = torch.as_tensor(inp_model.copy())

            # Định dạng lại tensor để phù hợp với mô hình
            num_inp, w, h, c = inp_model.shape
            inp_model = inp_model.permute(0, 3, 1, 2)
            inp_model = inp_model.reshape(num_inp * c, w, h)

            return inp_model, d_img, s_img, t_img
    
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


    def input_folder(self, data_dir, out_dir):
        """Xử lý toàn bộ thư mục ảnh."""
        os.makedirs(out_dir, exist_ok=True)
        img_folders = glob.glob(f"{data_dir}/*")

        for folder in tqdm(img_folders, desc="Processing folders"):
            img_files = glob.glob(os.path.join(folder, "*"))
            if not img_files:
                print(f"Thư mục {folder} không chứa ảnh!")
                continue

            number = os.path.basename(folder)
            temp_dir = os.path.join(folder, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            # Bước 1: Tạo ảnh điều chỉnh phơi sáng
            img_path = img_files[0]  # Giả định chỉ có 1 ảnh gốc
            img = cv2.imread(img_path)
            exposure_paths = self.apply_exposure_adjustment(img, temp_dir)

            # Bước 2: Tạo HDR từ 5 ảnh
            hdr_path = os.path.join(folder, f"{number}_2.png")
            self.create_hdr_image(exposure_paths, hdr_path)

            # Bước 3: Tạo ảnh sáng và tối từ HDR
            bright_path = os.path.join(folder, f"{number}_1.png")
            dark_path = os.path.join(folder, f"{number}_3.png")
            hdr_img = cv2.imread(hdr_path)
            self.adjust_exposure(hdr_img, [bright_path, dark_path])

            # Bước 4: Xử lý 3 ảnh đầu vào
            inp_model, d_img, s_img, t_img = self.input_3_images(bright_path, hdr_path, dark_path)

            # Chạy inference
            if self.onnx_model_path:
                raw_outputs = self.onnx_inference(inp_model)
                weights = raw_outputs[0]
            else:
                with torch.no_grad():
                    img_tensor = inp_model.to(dtype=torch.float32).unsqueeze(0)
                    _, weights = self.net(img_tensor)

            # Hậu xử lý weights
            
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

            # Kết hợp các ảnh đầu vào dựa trên weights
            for i in range(weights.shape[1]):
                if i == 0:
                    weights = weights.to(device)
                    out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
                else:
                    out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]

        # Lưu ảnh kết quả
        result = ops.to_image(out_img[0, :, :, :])
        result_folder = os.path.join(out_dir, str(number))
        os.makedirs(result_folder, exist_ok=True)
        result_path = os.path.join(result_folder, f"{number}_output.png")
        result.save(result_path)

    def gradio_process(self, img):
        """
        Xử lý pipeline inference cho giao diện Gradio.
        
        Args:
            img (PIL.Image): Ảnh đầu vào từ giao diện Gradio.

        Returns:
            PIL.Image: Ảnh đầu ra sau khi xử lý.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Lưu ảnh đầu vào vào thư mục tạm
        temp_folder = "temp_gradio"
        os.makedirs(temp_folder, exist_ok=True)
        img_path = os.path.join(temp_folder, "input.png")
        img.save(img_path)

        # Đọc ảnh và tạo ảnh điều chỉnh phơi sáng
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn {img_path}!")

        exposure_paths = self.apply_exposure_adjustment(img_cv, temp_folder)

        # Tạo HDR từ các ảnh điều chỉnh phơi sáng
        hdr_path = os.path.join(temp_folder, "hdr_image.png")
        self.create_hdr_image(exposure_paths, hdr_path)

        # Tạo ảnh sáng hơn và tối hơn từ HDR
        bright_path = os.path.join(temp_folder, "bright.png")
        dark_path = os.path.join(temp_folder, "dark.png")
        hdr_img = cv2.imread(hdr_path)
        self.adjust_exposure(hdr_img, [bright_path, dark_path])

        # Chuẩn bị 3 ảnh đầu vào cho mô hình
        inp_model, d_img, s_img, t_img = self.input_3_images(bright_path, hdr_path, dark_path)

        # Chạy inference
        if self.onnx_model_path:
            # inp_model = inp_model.cpu().numpy()
            raw_outputs = self.onnx_inference(inp_model)
            # weights = raw_outputs[0]
            weights = torch.from_numpy(raw_outputs[0]).float().to(device)
        else:
            with torch.no_grad():
                inp_tensor = inp_model.to(dtype=torch.float32).unsqueeze(0)
                _, weights = self.net(inp_tensor)

        # Hậu xử lý weights
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

        # Kết hợp các ảnh đầu vào dựa trên weights
        for i in range(weights.shape[1]):
            if i == 0:
                weights = weights.to(device)
                out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i].to(device)
            else:
                out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i].to(device)

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
