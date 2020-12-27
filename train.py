# -*-encoding:utf-8-*-
"""
# function/功能 : 
# @File : alexnet_main.py 
# @Time : 2020/7/22 8:55 
# @Author : kf
# @Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.utils.data
import tqdm
from torch import optim

from DataLoader import DogCatDataSet
from model import LeNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 权重初始化
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and classname.find('DeformConv2d') == -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


# 模型训练初始化
def model_init(temp_model):
    temp_model = nn.DataParallel(temp_model)
    temp_model.train(mode=True)
    temp_model.apply(weight_init)
    temp_model.to(device)
    return temp_model


# 获得数据
def get_dataset(data_dir, batch_size=64, n_threads=8, imageindex=0):
    dataset = DogCatDataSet(data_dir, imageindex=imageindex)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             drop_last=True, shuffle=True, num_workers=int(n_threads))
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    return dataloader

# 设置学习率调整策略
def set_learning_rate(optimizer, epoch):
    # if epoch<30:
    #     optimizer.param_groups[0]['lr'] = 1e-3
    # el
    if optimizer.param_groups[0]['lr']>1e-6:
        optimizer.param_groups[0]['lr'] = 1e-3 * 0.1 ** (epoch // 100)
    # optimizer.param_groups[0]['lr'] = 1e-4* epoch


def train():
    save_flag=True
    trainloader = get_dataset(r'H:/DataSet_All/猫狗识别/gpu/train', batch_size=64, imageindex=0)
    validationloader = get_dataset(r'H:/DataSet_All/猫狗识别/gpu/test', batch_size=64, imageindex=0)

    save_dir = 'output/'
    my_model = LeNet()
    my_model = model_init(my_model)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(my_model.parameters(), lr=0.001)
    epochs = 300
    # 训练

    for epoch in range(epochs):
        loss_list = []
        acc_list = []
        set_learning_rate(optimizer, epoch)

        learning_rate = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(trainloader, desc='train')
        tq.set_description('train Epoch{}    lr{}'.format(epoch, learning_rate))

        for images, labels in tq:
            images = images.to(device)
            labels = labels.to(device)
            outputs = my_model(images)
            loss = criterion(outputs, labels)

            my_model.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            loss_ave = sum(loss_list) / len(loss_list)

            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).sum().float() / labels.size(0)
            acc_list.append(accuracy)
            acc_ave = sum(acc_list) / len(acc_list)
            tq.set_postfix(loss="%.4f  accuracy:%.4f   loss_ave:%.5f  acc_ave:%.5f  "
                                % (loss.item(), accuracy, loss_ave, acc_ave))

        if save_flag:
            log = "\ntrain \tEpoch {}/{} \t Learning rate: {:.5f} \t Train loss_ave: {:.5f} \t  acc_ave: {:.5f} \t  " \
                   .format(epoch, epochs, learning_rate, loss_ave, acc_ave )
            # print(log)
            logFile = open(save_dir + '/log.txt', 'a')
            logFile.write(log + '\n')
            torch.save(my_model.state_dict(), save_dir + '/model_lastest.pt')

        if epoch % 1 == 0:
            loss_list = []
            acc_list = []
            with torch.no_grad():
                tq = tqdm.tqdm(validationloader, desc='teat')
                for images, labels in tq:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = my_model(images)
                    validation_loss = criterion(outputs, labels)
                    loss_list.append(validation_loss.item())
                    _, predicted = torch.max(outputs.data, 1)
                    accuracy = (predicted == labels).sum().float() / labels.size(0)
                    acc_list.append(accuracy)
                    acc_ave = sum(acc_list) / len(acc_list)
                    loss_ave = sum(loss_list) / len(loss_list)
                    tq.set_postfix(test_loss="%.4f  acc:%.4f   loss_ave:%.5f  acc_ave:%.5f  "
                                             % (validation_loss, accuracy, loss_ave, acc_ave))
                log = "\ntest \tEpoch {}/{} \t Learning rate: {:.5f} \t Train loss_ave: {:.5f} \t  acc_ave: {:.5f} \t  " \
                    .format(epoch, epochs, learning_rate, loss_ave, acc_ave)
                # print(log)
                logFile = open(save_dir + '/log.txt', 'a')
                logFile.write(log + '\n')


if __name__ == '__main__':
    train()
