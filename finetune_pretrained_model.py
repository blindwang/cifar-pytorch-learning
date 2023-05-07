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
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
    elif model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
    elif model_name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 10)
    elif model_name == "vgg16bn":
        model = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 10)
    elif model_name == "vgg19":
        model = models.vgg19(weights=VGG19_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 10)
    elif model_name == "vgg19bn":
        model = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 10)
    elif model_name == "alexnet":
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 10)
    elif model_name == "googlenet":
        model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

    # 对于模型的每个权重，使其不进行反向传播，即固定参数
    for param in model.parameters():
        param.requires_grad = False
    # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层
    if model_name in ["vgg16", "vgg19", "vgg16bn", "vgg19bn", "alexnet"]:
        for param in model.classifier[6].parameters():
            param.requires_grad = True
    else:
        for param in model.fc.parameters():
            param.requires_grad = True

    return model

