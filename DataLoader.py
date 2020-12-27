#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image
from torchvision import transforms
import random

# 数据集读取
class DogCatDataSet(torch.utils.data.Dataset):
    def __init__(self, img_dir,imageindex=0):
        self.allFilePath = self.getFiles(img_dir, '.jpg')
        if imageindex!=0:
            self.allFilePath = random.sample(self.allFilePath, imageindex)
        data_transform = transforms.Compose([
            transforms.Resize(256),  # resize到256
            transforms.CenterCrop(224),  # crop到224
            transforms.ToTensor(),
            # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor /255.操作

        ])
        self.transform = data_transform

    def getFiles(self, dir, suffix):  # 查找根目录，文件后缀
        res = []
        for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
            for filename in files:
                name, suf = os.path.splitext(filename)  # =>文件名,文件后缀
                if suf == suffix:
                    res.append(os.path.join(root, filename))  # =>吧一串字符串组合成路径
        return res

    # 作为迭代器必须要有的
    def __getitem__(self, index):
        img_path = self.allFilePath[index]
        label = 1 if 'dog' in os.path.split(img_path)[1] else 0  # 狗的label设为1，猫的设为0
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.allFilePath)


#  读取数据

if __name__ == "__main__":

    CLASSES = {0: "cat", 1: "dog"}
    img_dir = r"H:\DataSet_All\猫狗识别\cpu\train"
    dataSet = DogCatDataSet(img_dir=img_dir)
    dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=8, shuffle=True, num_workers=4)
    image_batch, label_batch = iter(dataLoader).next()
    for i in range(image_batch.data.shape[0]):
        label = np.array(label_batch.data[i])  ## tensor ==> numpy
        # print(label)
        img = np.array(image_batch.data[i] * 255, np.int32)
        print(CLASSES[int(label)])
        plt.imshow(np.transpose(img, [2, 1, 0]))
        plt.show()
