import numpy as np
import torch
import torch.optim as optim

from torchvision import models
from torchvision.models import VGG19_Weights

vgg19_net = models.vgg19(weights=VGG19_Weights.DEFAULT).features

# style_weight = 1e3
# style_weight = 1e9

layers = {'0': 'conv1_1',
          '5': 'conv2_1',
          '10': 'conv3_1', 
          '19': 'conv4_1',
          '21': 'conv4_2',
          '28': 'conv5_1'}

gt_weights = {'conv1_1': 1.0,
              'conv2_1': 0.6,
              'conv3_1': 0.4,
              'conv4_1': 0.3,
              'conv5_1': 0.1}

def get_features(img, net, layers):
    features = {}
    x = img

    for name, layer in net._modules.items():
        x = layer(x)

        if name in layers:
            features[layers[name]] = x

    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    
    return gram