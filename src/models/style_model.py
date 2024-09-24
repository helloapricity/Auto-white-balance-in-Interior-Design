import torch
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from collections import namedtuple

class StyleModel(torch.nn.Module):
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True, progress=show_progress).features
        
        return_layers = {
            '3': 'relu1_2',     # Corresponds to relu1_2
            '8': 'relu2_2',     # Corresponds to relu2_2
            '15': 'relu3_3',    # Corresponds to relu3_3
            '22': 'relu4_3'     # Corresponds to relu4_3
        }
        
        self.model = IntermediateLayerGetter(vgg_pretrained_features, return_layers=return_layers)
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.content_feature_maps_index = 1  # relu2_2
        self.style_feature_maps_indices = list(range(len(self.layer_names)))  # all layers used for style representation

        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        # print(out['relu1_2'].shape)
        # print(out['relu2_2'].shape)
        # print(out['relu3_3'].shape)
        # print(out['relu4_3'].shape)
        out = vgg_outputs(out['relu1_2'], out['relu2_2'], out['relu3_3'], out['relu4_3'])
        return out

# if __name__ == '__main__':
#     IMG_SIZE = 224
#     img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
#     model = StyleModel()  
#     param = model(img)
