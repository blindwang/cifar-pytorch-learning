import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np

from train import train
from evaluate import evaluate
from dataloader import build_dataloader
from model import ResNet, VGG, GoogleNet, LeNet5, AlexNet
from utils import bulid_tensorboard_writer
from earlystop import EarlyStopping

"""随机种子"""
seed = 2023

# Python随机数生成器的种子
random.seed(seed)

# Numpy随机数生成器的种子
np.random.seed(seed)

# Pytorch随机数生成器的种子
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 判断是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""网络"""
models_ls = {"resnet18": ResNet(), "vgg": VGG(), "alexnet": AlexNet(), "googlenet": GoogleNet(), "lenet": LeNet5()}

"""定义优化器列表"""
optimizers = [
    {'name': 'SGD', 'optimizer': optim.SGD, 'lr': 0.01},
    {'name': 'Adam', 'optimizer': optim.Adam, 'lr': 0.01},
    {'name': 'RMSprop', 'optimizer': optim.RMSprop, 'lr': 0.01},
    {'name': 'Adadelta', 'optimizer': optim.Adadelta, 'lr': 0.01},
    {'name': 'Adagrad', 'optimizer': optim.Adagrad, 'lr': 0.01},
]

"""默认超参数"""
batch_size = 128
learning_rate = 0.001
num_epochs = 100
num_workers =  2 # CPU中为0，GPU可以不为0

"""默认dataloader"""
trainloader, testloader, trainset, testset = build_dataloader(batch_size, num_workers)


for model_name, model in models_ls.items():
    for opt in optimizers:
        net = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = opt['optimizer'](net.parameters(), lr=opt['lr'])
        earlystop = EarlyStopping(model_name=model_name)

        print(f"Training {model_name}... w/ {opt['name']}")

        """tensorboard writer"""
        train_summary_writer, test_summary_writer = bulid_tensorboard_writer(f"compare_optim", opt['name'])

        # 训练模型
        for epoch in range(num_epochs):
            train(trainloader, net, criterion, optimizer, epoch, device, train_summary_writer)
            evaluate(testloader, net, epoch, device, test_summary_writer)
            acc = evaluate(testloader, net, epoch, device, test_summary_writer)
            earlystop(-acc, model_name+"_hand")
            if earlystop.early_stop:
                break

        print('Finished Training')
