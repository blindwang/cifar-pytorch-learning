import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

from train import train
from evaluate import evaluate
from dataloader import build_dataloader
from finetune_pretrained_model import finetune_pretrained_model
from model import LeNet5
from utils import bulid_tensorboard_writer
from optimizer import get_optim, get_scheduler
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

# models_ls = ["resnet18", "resnet50", "vgg16", "vgg16bn", "vgg19",
#              "vgg19bn", "alexnet", "googlenet"]

models_ls = ["lenet", "alexnet", "resnet18", "vgg16", "googlenet"]

# 判断是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

for model_name in models_ls[1:]:
    print(f"Training {model_name} model...")

    """默认超参数"""
    batch_size = 128
    learning_rate = 5e-6
    num_epochs = 20
    num_workers = 2  # CPU中为0，GPU可以不为0

    """tensorboard writer"""
    train_summary_writer, test_summary_writer = bulid_tensorboard_writer("compare_base", model_name)

    """默认dataloader"""
    trainloader, testloader, trainset, testset = build_dataloader(batch_size, num_workers)

    """定义默认网络"""
    if model_name == 'lenet':
        model = LeNet5().to(device)
        learning_rate = 0.001
    else:
        model = finetune_pretrained_model(model_name).to(device)

    """定义默认损失函数和优化器"""
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = get_optim(model, model_name, optim_name="adam", lr=learning_rate,
                          weight_decay=0.05, lr_decay_factor=0.75)
    scheduler = get_scheduler(optimizer,
                              3 * math.ceil(len(trainset) / batch_size),
                              num_epochs * math.ceil(len(trainset) / batch_size))

    """设置早停策略"""
    earlystop = EarlyStopping(patience=3, model_name=model_name)

    # 训练模型
    for epoch in range(num_epochs):
        train(trainloader, model, criterion, optimizer, epoch, device, train_summary_writer, scheduler)
        acc = evaluate(testloader, model, epoch, device, test_summary_writer)
        earlystop(-acc, model)
        if earlystop.early_stop:
            break

    print('Finished Training')
