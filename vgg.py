import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

IMAGE_MEAN = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1) / 255.0

def preprocess(image):
    return image - IMAGE_MEAN.to(image.device)

def vgg19_features(device='cuda'):
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    layers = {
        'relu1_1': 1,
        'relu1_2': 3,
        'relu2_1': 6,
        'relu2_2': 8,
        'relu3_1': 11,
        'relu3_2': 13,
        'relu3_3': 15,
        'relu3_4': 17,
        'relu4_1': 20,
        'relu4_2': 22,
        'relu4_3': 24,
        'relu4_4': 26,
        'relu5_1': 29,
        'relu5_2': 31,
        'relu5_3': 33,
        'relu5_4': 35
    }

    class VGGFeatures(nn.Module):
        def __init__(self, target_layers):
            super(VGGFeatures, self).__init__()
            self.vgg = vgg
            self.target_layers = {k: v for k, v in layers.items() if k in target_layers}

        def forward(self, x):
            features = {}
            for name, module in self.vgg._modules.items():
                x = module(x)
                for key, value in self.target_layers.items():
                    if int(name) == value:
                        features[key] = x
            return features

    return VGGFeatures
