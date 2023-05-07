import torch.nn as nn
import torch
import torch.optim as optim
import random
import math
import numpy as np

from train import train
from evaluate import evaluate
from dataloader import build_dataloader
from dataloader_simple import build_dataloader_simple
from model import LeNet5
from finetune_pretrained_model import finetune_pretrained_model
from utils import bulid_tensorboard_writer
from earlystop import EarlyStopping
from optimizer import get_optim, get_scheduler

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
# models_ls = {"resnet18": ResNet(), "vgg": VGG(), "alexnet": AlexNet(), "googlenet": GoogleNet(), "lenet": LeNet5()}

models_ls = ["lenet", "alexnet", "resnet18", "vgg16", "googlenet"]

"""定义优化器列表"""
optimizers = [
    {'name': 'SGD', 'optimizer': optim.SGD},
    {'name': 'Adam', 'optimizer': optim.Adam},
    {'name': 'RMSprop', 'optimizer': optim.RMSprop},
    {'name': 'Adadelta', 'optimizer': optim.Adadelta},
    {'name': 'Adagrad', 'optimizer': optim.Adagrad},
]

"""默认超参数"""
batch_size = 128
learning_rate = 0.001
num_epochs = 20
num_workers = 2  # CPU中为0，GPU可以不为0

for model_name in models_ls:
    """读取数据"""
    if model_name == 'lenet':
        trainloader, testloader, trainset, testset = build_dataloader_simple(batch_size, num_workers)
    else:
        trainloader, testloader, trainset, testset = build_dataloader(batch_size, num_workers)

    for opt in optimizers:
        if model_name == 'lenet':
            model = LeNet5().to(device)
            learning_rate = 0.001
        else:
            model = finetune_pretrained_model(model_name=model_name).to(device)
            learning_rate = 5e-6
        criterion = nn.CrossEntropyLoss()
        optimizer = opt['optimizer'](model.parameters(), lr=learning_rate)
        optimizer = get_optim(model, model_name, optim_name="adam", lr=learning_rate,
                              weight_decay=0.05, lr_decay_factor=0.75)
        scheduler = get_scheduler(optimizer,
                                  3 * math.ceil(len(trainset) / batch_size),
                                  num_epochs * math.ceil(len(trainset) / batch_size))
        earlystop = EarlyStopping(patience=3, model_name=model_name + '_' + opt['name'])

        print(f"Training {model_name}... w/ {opt['name']}")

        """tensorboard writer"""
        train_summary_writer, test_summary_writer = bulid_tensorboard_writer(f"compare_optim/{model_name}", opt['name'])

        # 训练模型
        for epoch in range(num_epochs):
            train(trainloader, model, criterion, optimizer, epoch, device, train_summary_writer, scheduler)
            acc = evaluate(testloader, model, epoch, device, test_summary_writer)
            earlystop(-acc, model)
            if earlystop.early_stop:
                break

        print('Finished Training')
