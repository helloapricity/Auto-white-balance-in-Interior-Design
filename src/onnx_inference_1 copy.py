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
        
        # Nếu có đường dẫn đến model ONNX, khởi tạo phiên làm việc ONNX
        if onnx_model_path:
            self.onnx_session = ort.InferenceSession(onnx_model_path)
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            self.onnx_output_name = self.onnx_session.get_outputs()[0].name

    def adjust_exposure(self, image, gamma_values=None, linear_factors=None):
        """Điều chỉnh độ phơi sáng của ảnh để tạo ra phiên bản tối và sáng."""
        if gamma_values is None:
            gamma_values = [0.8, 1.2]  # Gamma mặc định cho ảnh tối và sáng
        if linear_factors is None:
            linear_factors = [0.8, 1.2]  # Scaling mặc định

        adjusted_images = []
        for gamma, factor in zip(gamma_values, linear_factors):
            # Gamma correction
            gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            adjusted_image = cv2.LUT(image, gamma_table)

            # Linear scaling
            adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=factor, beta=0)
            adjusted_images.append(adjusted_image)

        return adjusted_images

    def preprocess_image(self, image):
        """
        Xử lý ảnh để tạo batch gồm ảnh gốc và 2 ảnh đã điều chỉnh phơi sáng.
        """
        # Resize ảnh về kích thước yêu cầu
        if image.shape[:2] != (self.t_size, self.t_size):
            image = cv2.resize(image, (self.t_size, self.t_size))

        # Tạo ảnh phơi sáng
        adjusted_images = self.adjust_exposure(image)

        # Combine ảnh gốc và ảnh điều chỉnh thành batch
        images = [image] + adjusted_images
        images_tensor = np.stack([img.transpose(2, 0, 1) for img in images])  # Chuyển sang định dạng CHW
        return images_tensor.astype(np.float32) / 255.0  # Normalize ảnh

    def process_image(self, image_path, output_path):
        """
        Xử lý pipeline chính để xử lý 1 ảnh đầu vào.
        """
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn {image_path}!")

        # Tiền xử lý ảnh
        input_images = self.preprocess_image(image)

        # Chạy inference
        input_images = np.expand_dims(input_images, axis=0)  # Thêm batch dimension
        if self.onnx_model_path:
            outputs = self.onnx_session.run([self.onnx_output_name], {self.onnx_input_name: input_images})
            result = outputs[0]
        else:
            raise NotImplementedError("Hỗ trợ ONNX model là bắt buộc trong pipeline này.")

        # Lưu ảnh đầu ra
        cv2.imwrite(output_path, result[0].transpose(1, 2, 0) * 255)

    def process_folder(self, data_dir, out_dir):
        """
        Xử lý toàn bộ thư mục ảnh.  
        """
        os.makedirs(out_dir, exist_ok=True)
        image_paths = glob.glob(f"{data_dir}/*.jpg")  # Thay đổi định dạng file nếu cần

        for image_path in tqdm(image_paths):
            base_name = os.path.basename(image_path).split('.')[0]
            output_path = os.path.join(out_dir, f"{base_name}_output.png")
            self.process_image(image_path, output_path)

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
    
    def copy_subfolders(self, source_dir, destination_dir):
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        items = os.listdir(source_dir)
        for item in items:
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path):
                destination_path = os.path.join(destination_dir, item)
                shutil.copytree(item_path, destination_path, dirs_exist_ok=True)

    def rename_and_save(self, img_path, save_dir, new_name):
        """
        Đổi tên ảnh gốc thành `new_name` và lưu lại.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {img_path}")
        os.makedirs(save_dir, exist_ok=True)
        new_path = os.path.join(save_dir, new_name)
        cv2.imwrite(new_path, img)
        return new_path

    def create_augmented_images(self, img_path):
        """
        Tạo ra hai ảnh mới bằng cách chỉnh sửa phơi sáng hoặc áp dụng filter.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {img_path}")

        # Tạo hai phiên bản ảnh với gamma khác nhau
        gamma_values = [0.5, 1.5]
        augmented_images = []
        for i, gamma in enumerate(gamma_values, start=2):
            gamma_table = np.array([((j / 255.0) ** gamma) * 255 for j in np.arange(256)]).astype("uint8")
            adjusted_img = cv2.LUT(img, gamma_table)
            aug_name = img_path.replace("_1.png", f"_{i}.png")
            cv2.imwrite(aug_name, adjusted_img)
            augmented_images.append(aug_name)
        return augmented_images
    
    def input_folder(self, data_dir, out_dir):
        """
        Xử lý toàn bộ thư mục ảnh, bao gồm đổi tên ảnh, tạo ảnh bổ sung, 
        chạy inference, và lưu kết quả đầu ra.

        Args:
            data_dir (str): Thư mục chứa các ảnh đầu vào.
            out_dir (str): Thư mục để lưu kết quả đầu ra.
        """
        os.makedirs(out_dir, exist_ok=True)

        # Duyệt qua tất cả các thư mục con trong `data_dir`
        img_folders = glob.glob(f'{data_dir}/*')
        for folder in tqdm(img_folders, desc="Processing image folders"):
            img_files = glob.glob(os.path.join(folder, '*'))
            if not img_files:
                print(f"Thư mục {folder} không chứa ảnh nào, bỏ qua.")
                continue

            number = os.path.basename(folder)  # Lấy ID của thư mục (ví dụ: 1, 2, ...)
            base_name = os.path.basename(folder)

            # Đổi tên ảnh gốc thành <id>_1.png
            img1_path = os.path.join(folder, f"{number}_1.png")
            if not os.path.exists(img1_path):
                self.rename_and_save(img_files[0], folder, f"{number}_1.png")

            # Tạo 2 ảnh bổ sung <id>_2.png và <id>_3.png
            img2_path, img3_path = self.create_augmented_images(img1_path)

            # Danh sách các ảnh đầu vào và nhãn
            inps = [img1_path, img2_path, img3_path]
            labels = [os.path.join(folder, f"{number}_G.jpg")]

            print(f"Processing folder: {base_name}")
            print(f"Input images: {inps}")
            print(f"Labels: {labels}")

            # Xử lý 3 ảnh đầu vào để chuẩn bị input cho mô hình
            inp_model, d_img, s_img, t_img = self.input_3_images(inps[0], inps[1], inps[2])

            # Chạy ONNX inference
            if self.onnx_model_path:
                raw_outputs = self.onnx_inference(inp_model)
                weights = raw_outputs[0]  # Giả sử đầu ra là weights
                print("Weights shape from ONNX:", weights.shape)
            else:
                # Nếu không sử dụng ONNX, fallback sang mô hình PyTorch
                with torch.no_grad():
                    img = inp_model.to(device=device, dtype=torch.float32).unsqueeze(0)
                    _, weights = self.net(img)
                print("Weights shape from PyTorch:", weights.shape)

            # Xử lý weights để chuẩn bị kết hợp ảnh
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
                    out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]

            # Chuyển ảnh kết quả thành định dạng lưu được và lưu vào thư mục output
            result = ops.to_image(out_img[0, :, :, :])
            result_folder = os.path.join(out_dir, base_name)
            os.makedirs(result_folder, exist_ok=True)
            result_path = os.path.join(result_folder, f"{base_name}_output.png")
            result.save(result_path)

        print(f"Hoàn thành xử lý tất cả các thư mục trong {data_dir}. Kết quả lưu tại: {out_dir}")


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
