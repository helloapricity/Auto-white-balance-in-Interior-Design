from models.wb_net import WBNet
import torch 
import torch.onnx 
from arguments import get_args
from utils.ops import get_sobel_kernel
from models.vgg19 import vgg19_net
from models.litawb_style_module import LitAWBStyleLoss
import os
from test import attem_load_author
args = get_args()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = WBNet(norm=args.norm, inchnls=3 * len(args.wb_settings))
x_kernel, y_kernel = get_sobel_kernel(chnls=len(args.wb_settings))
litmodel = LitAWBStyleLoss(model=net, lr=args.lr, smooth_weight=args.smoothness_weight, x_kernel=x_kernel, y_kernel=y_kernel, vgg_model=vgg19_net)
epoch = "141"
model = f"sample-epoch={epoch}"
# model_path = f"output/{model}.ckpt"
model_path = os.path.join(os.path.dirname(__file__), "..", "output", f"{model}.ckpt")
# model_path = f"/home/tiennv/FPT/training_wb/weights/WB_model_p_64_D_S_T.pth"

checkpoint = torch.load(model_path)

litmodel = attem_load_author(litmodel, model_path)
# litmodel.to(device=device)




dummy_input = torch.randn(1, 9 , 320, 320, dtype=torch.float32)
onnx_path = "wbnet_model.onnx"
torch.onnx.export(
    litmodel,                 # Mô hình PyTorch
    dummy_input,           # Dummy input
    onnx_path,             # Đường dẫn lưu file ONNX
    export_params=True,    # Lưu trọng số trong file ONNX
    opset_version=12,      # Phiên bản ONNX opset (có thể tăng nếu cần)
    do_constant_folding=True,  # Áp dụng constant folding
    input_names=['input'],     # Tên đầu vào
    output_names=['output'],   # Tên đầu ra
    dynamic_axes={             # Trường hợp hỗ trợ batch size linh hoạt
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print(f"Model exported to {onnx_path}")