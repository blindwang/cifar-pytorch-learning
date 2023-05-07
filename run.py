import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from train import train
from evaluate import evaluate
from dataloader import build_dataloader
from finetune_pretrained_model import finetune_pretrained_model
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

# models_ls = ["resnet18", "resnet50", "vgg16", "vgg16bn", "vgg19",
#              "vgg19bn", "alexnet", "googlenet"]

models_ls = ["resnet18", "vgg16", "googlenet"]

# 判断是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

for model_name in models_ls[2:]:
    print(f"Training {model_name} model...")

    """默认超参数"""
    batch_size = 128
    learning_rate = 1e-5
    num_epochs = 100
    num_workers = 2  # CPU中为0，GPU可以不为0

    """tensorboard writer"""
    train_summary_writer, test_summary_writer = bulid_tensorboard_writer("compare_base", model_name)

    """默认dataloader"""
    trainloader, testloader, trainset, testset = build_dataloader(batch_size, num_workers)

    """定义默认网络"""
    # net = Net().to(device)
    model = finetune_pretrained_model(model_name).to(device)

    """定义默认损失函数和优化器"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    """设置早停策略"""
    earlystop = EarlyStopping(model_name=model_name)

    # 训练模型
    for epoch in range(num_epochs):
        train(trainloader, model, criterion, optimizer, epoch, device, train_summary_writer)
        acc = evaluate(testloader, model, epoch, device, test_summary_writer)
        earlystop(-acc, model_name)
        if earlystop.early_stop:
            break

    print('Finished Training')
