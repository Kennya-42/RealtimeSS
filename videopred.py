import cv2
from PIL import Image
from collections import OrderedDict
from models.erfnet import ERFNet
import torchvision.transforms as transforms
import torchvision
import utils
import os
import numpy as np
import time
import torch
#python3 main.py --mode vidpred --save-dir save/ERFnet/ERFnet_aug/ --width 1920 --height 1080
def vidpred(args):
    start_time = time.time()
    vidcap = cv2.VideoCapture('test_content/testvid.webm')
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (args.width,args.height),True)
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
    model = ERFNet(len(class_encoding)).cuda()
    checkpoint = torch.load(os.path.join(args.save_dir, args.name))
    model.load_state_dict(checkpoint['state_dict'])
    for j in range(500):
        if j%10 ==0:
            print(j)
        images = get_batch(args.batch_size,vidcap,args)
        if images is None:
            break
        with torch.no_grad():
            images = images.cuda()
            predictions = model(images)
            del images
        _, predictions = torch.max(predictions.data, 1)
        label_to_rgb = transforms.Compose([utils.LongTensorToRGBPIL(class_encoding),transforms.ToTensor()])
        lb = utils.batch_transform(predictions.cpu(), label_to_rgb)
        del predictions
        #cast as numpy from tensor
        outpred = lb.numpy()
        #move color channel to back for cv2
        outpred = np.moveaxis(outpred, 1, -1)
        outpred = (outpred*255).astype(np.uint8)
        for i in range(args.batch_size):
            cv2.imshow('image',outpred[i,:,:,:])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            out.write(outpred[i,:,:,:])
    elapsed_time = time.time() - start_time
    print('Time: ',elapsed_time)

def get_batch(batchsize,vidcap,args):
    index = 0
    images = []
    while index < batchsize:
        # print(index)
        index += 1
        success,img = vidcap.read()
        if not success:
            return None
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((args.width, args.height), Image.BICUBIC)
        img = np.array(img)
        img = np.moveaxis(img, 2, 0)
        images.append(img)
    images = np.array(images)
    images = torch.from_numpy(images).type(torch.FloatTensor)
    return images
