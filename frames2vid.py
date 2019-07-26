import cv2
from PIL import Image
from collections import OrderedDict
from models.erfnet import ERFNet
from models.deeplab import DeepLab
import torchvision.transforms as transforms
import torchvision
import utils
import os
import numpy as np
import time
import glob
import torch

import matplotlib.pyplot as plt

def frames2vid(args):
    start_time = time.time()
    out = cv2.VideoWriter('predvideo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (args.width,args.height*2),True)
    class_encoding = OrderedDict([
                        ('unlabeled', (0, 0, 0)),
                        ('road', (128, 64, 128)),
                        ('sidewalk', (244, 35, 232)),
                        ('building', (70, 70, 70)),
                        ('wall', (102, 102, 156)),
                        ('fence', (190, 153, 153)),
                        ('pole', (153, 153, 153)),
                        ('traffic_light', (250, 170, 30)),
                        ('traffic_sign', (220, 220, 0)),
                        ('vegetation', (107, 142, 35)),
                        ('terrain', (152, 251, 152)),
                        ('sky', (70, 130, 180)),
                        ('person', (220, 20, 60)),
                        ('rider', (255, 0, 0)),
                        ('car', (0, 0, 142)),
                        ('truck', (0, 0, 70)),
                        ('bus', (0, 60, 100)),
                        ('train', (0, 80, 100)),
                        ('motorcycle', (0, 0, 230)),
                        ('bicycle', (119, 11, 32)) ])
    if args.model.lower() == 'erfnet':
        print("Model Arch: ERFnet")
        model = ERFNet(num_classes).cuda()
    else:
        print("Model Arch: DeeplabV3+")
        model = DeepLab(backbone='resnet', output_stride=8, num_classes=20,freeze_bn=True).cuda()

    checkpoint = torch.load(os.path.join(args.save_dir, args.name))
    model.load_state_dict(checkpoint['state_dict'])
    files = glob.glob('test_content/stuttgart_02/*.png')
    files.sort()
    for j in range(len(files)):
        if j%1 ==0:
            print(j)
            img = Image.open(files[j])
            img = img.resize((args.width, args.height), Image.BICUBIC)
            images = transforms.ToTensor()(img)
            torch.reshape(images, (1, 3, args.width, args.height))
            images= images.unsqueeze(0)
            images = images.cuda()
            with torch.no_grad():
                predictions = model(images)
            _, predictions = torch.max(predictions.data, 1)
            predictions = utils.LongTensorToRGBPIL(class_encoding)(predictions.cpu())
            predictions = np.asarray(predictions)
            predictions = cv2.cvtColor(predictions,cv2.COLOR_RGB2BGR)
            img = np.asarray(img)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            frame = np.vstack((img,predictions))
            out.write(frame)
            

def get_batch(batchsize,files):
    index = 0
    images = []
    while index < batchsize:
        img = cv2.imread(files[index],1)
        index += 1
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        images.append(img)
   
    return images
