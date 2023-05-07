# import tensorflow as tf
from tqdm import tqdm
from tensorboardX import SummaryWriter


def train(trainloader, net, criterion, optimizer, epoch, device, train_summary_writer):
    running_loss = 0.0
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
            running_loss += loss.item()
            # 在tqdm中展示loss
            tbar.set_postfix(running_loss=loss.item())
            # 更新进度条
            tbar.update()

            if (i + 1) % 100 == 0:
                running_loss = 0.0

            # with train_summary_writer.as_default():
            #     tf.summary.scalar('loss', loss.item(), step=epoch * len(trainloader) + i)
            train_summary_writer.add_scalar('loss', loss.item(), epoch * len(trainloader) + i)
