import torchvision.models as models
from torch import optim
from torch.optim import lr_scheduler
from torchvision.models import ResNet18_Weights, ResNet50_Weights, VGG16_Weights, VGG16_BN_Weights, VGG19_Weights, \
    VGG19_BN_Weights, AlexNet_Weights, GoogLeNet_Weights
import torch.nn as nn
import torch


def finetune_pretrained_model(model_name):
    if model_name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    elif model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model_name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    elif model_name == "vgg16bn":
        model = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
    elif model_name == "vgg19":
        model = models.vgg19(weights=VGG19_Weights.DEFAULT)
    elif model_name == "vgg19bn":
        model = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
    elif model_name == "alexnet":
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
    elif model_name == "googlenet":
        model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 10.
    model.fc = nn.Linear(num_ftrs, 10)
    return model

