import timm
import torch
import torch.nn as nn

class StyleModel(nn.Module):
    def __init__(self, freeze=True):  # Thêm tham số 'freeze' để quyết định có đóng băng tham số hay không
        super(StyleModel, self).__init__()
        self.encode = timm.create_model('vgg16', pretrained=True)
        
        # Initialize layers using nn.ModuleList
        self.layers = nn.ModuleList(self.create_layers())
        
        # print(f"StyleModel initialized with freeze={freeze}")
        
        if freeze:
            # Đóng băng các tham số của mô hình
            for param in self.encode.parameters():
                param.requires_grad = False
        

    def create_layers(self):
        # Extract the children layers of the model
        children = list(self.encode.children())[0]

        # Store layers in a list
        layers = []
        start = 0
        # 2, 7, 14, 21
        for end in [2, 7, 14, 21]:
            layers.append(nn.Sequential(*children[start:end]))
            start = end
        return layers

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        
        if x.dim() == 5:
            x = x.view(-1, *x.shape[-3:])  # Reshape to [batch_size, channels, height, width]

        outputs = []
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
            outputs.append(x)
        return outputs

# if __name__ == '__main__':
#     IMG_SIZE = 224
#     img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
#     model = StyleModel(freeze=True)  # Tạo mô hình và đóng băng các tham số
#     param = model(img)
#     print(len(param))
