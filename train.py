# import tensorflow as tf
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter


def train(trainloader, net, criterion, optimizer, epoch, device, train_summary_writer, scheduler):
    net.train()
    with tqdm(trainloader, desc=f'Train epoch {epoch}') as tbar:
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # 调节学习率
            # accuracy
            accuracy = (labels == torch.argmax(outputs, dim=1)).float().sum() / labels.shape[0]
            # 在tqdm中展示loss
            tbar.set_postfix(running_loss=loss.item(), acc=accuracy)
            # 更新进度条
            tbar.update()

            # with train_summary_writer.as_default():
            #     tf.summary.scalar('loss', loss.item(), step=epoch * len(trainloader) + i)
            train_summary_writer.add_scalar('loss', loss.item(), epoch * len(trainloader) + i)
