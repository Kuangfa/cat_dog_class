import torch
import torch.utils.data
from PIL import Image, ImageDraw, ImageFont
from numpy import unicode
from torchvision import transforms

from model import *

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

import cv2
def drawText(imagepath, text, savepath):
    # 图像从OpenCV格式转换成PIL格式
    # img_PIL = Image.fromarray(image)
    # font = ImageFont.load_default()
    # # 需要先把输出的中文字符转换成Unicode编码形式
    # if not isinstance(text, unicode):
    #     text = text.decode('utf-8')
    # draw = ImageDraw.Draw(img_PIL)
    # draw.text((0, 0), text, fill=(255,0,0), font=font)
    # # 使用PIL中的save方法保存图片到本地
    # img_PIL.save(imagepath)
    img = cv2.imread(imagepath)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    cv2.putText(img, text, (10, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 5,bottomLeftOrigin=False)
    cv2.imwrite(savepath,img)


def read_imagepath(image_path):
    image = Image.open(image_path).convert('RGB')
    data_transform = transforms.Compose([
        transforms.Resize(256),  # resize到256
        transforms.CenterCrop(224),  # crop到224
        transforms.ToTensor(),
        # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor /255.操作

    ])
    image = data_transform(image)
    return image


import os
import numpy

def test(imagepath):
    assert os.path.exists(imagepath)
    CLASSES = {0: "cat", 1: "dog"}
    save_dir = 'output/'
    my_model = LeNet()
    my_model = model_init(my_model)
    # 训练
    if torch.cuda.is_available():
        my_model.load_state_dict(torch.load(save_dir + 'model_lastest.pt'))
    else:
        my_model.load_state_dict(
            torch.load(save_dir + 'model_lastest.pt', map_location='cpu'))

    with torch.no_grad():
        image = read_imagepath(imagepath).unsqueeze(0)
        image = image.to(device)
        outputs = my_model(image)
        _, predicted = torch.max(outputs.data, 1)
        label=CLASSES[int(predicted.item())]
        print(label)
        text='label:'+label
        drawText(imagepath,text,save_dir+os.path.split(imagepath)[1])


if __name__ == '__main__':
    imagepath = 'sample/dog.1639.jpg'
    test(imagepath)
