import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from lr_scheduler import Cust_LR_Scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict
from models.enet import ENet
from models.erfnet import ERFNet
from models.deeplab import DeepLab
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
import utils
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from videopred import vidpred
from frames2vid import frames2vid

# Get the arguments
args = get_arguments()
assert torch.cuda.is_available(), "no GPU connected"

def load_dataset(dataset):
    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    train_set = dataset(root_dir=args.dataset_dir, mode='train', height=args.height, width=args.width)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_set = dataset(root_dir=args.dataset_dir, mode='val',height=args.height, width=args.width)
    val_loader = data.DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers)
    test_loader = val_loader

    class_encoding = train_set.color_encoding
    num_classes = len(class_encoding)
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))
    #Get a batch of samples to display
    # for i in range(len(train_set)):
    timages, tlabels = iter(train_loader).next()
    # for i in range(len(val_set)): 
    vimages, vlabels = iter(val_loader).next()
    print("Train Image size:", timages.size())
    print("Train Label size:", tlabels.size())
    print("Val Image size:", vimages.size())
    print("Val Label size:", vlabels.size())
    print("Weighing technique:", args.weighing)
    # print("Computing class weights...") 
    # if args.weighing.lower() == 'enet':
    #     class_weights = utils.enet_weighing(train_loader, num_classes)
    # elif args.weighing.lower() == 'mfb':
    #     class_weights = utils.median_freq_balancing(train_loader, num_classes)
    # else:
    #     class_weights = None
    # class_weights = np.array([ 0.0000,  3.9490, 13.2085,  4.2485, 36.9267, 34.0329, 30.3585, 44.1654,
    #     38.5243,  5.7159, 32.2182, 16.3313, 30.7760, 46.8776, 11.1293, 44.1730,
    #     44.8546, 44.9209, 47.9799, 41.5301])
    class_weights = np.ones(num_classes)
    class_weights[0] = 2.8149201869965	
    class_weights[1] = 6.9850029945374	
    class_weights[2] = 3.7890393733978	
    class_weights[3] = 9.9428062438965	
    class_weights[4] = 9.7702074050903	
    class_weights[5] = 9.5110931396484	
    class_weights[6] = 10.311357498169	
    class_weights[7] = 10.026463508606	
    class_weights[8] = 4.6323022842407	
    class_weights[9] = 9.5608062744141	
    class_weights[10] = 7.8698215484619	
    class_weights[11] = 9.5168733596802	
    class_weights[12] = 10.373730659485	
    class_weights[13] = 6.6616044044495	
    class_weights[14] = 10.260489463806	
    class_weights[15] = 10.287888526917	
    class_weights[16] = 10.289801597595	
    class_weights[17] = 10.405355453491	
    class_weights[18] = 10.138095855713	
    class_weights[19] = 0
    print(class_weights)
    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float()
        # if args.ignore_unlabeled:
        #     ignore_index = list(class_encoding).index('unlabeled')
        #     class_weights[ignore_index] = 0
    
    # print("Class weights:", class_weights)
    return (train_loader, val_loader, test_loader), class_weights, class_encoding 
    
def train(train_loader, val_loader, class_weights, class_encoding):
    print("Training...")
    num_classes = len(class_encoding)
    # pick model
    # print("Loading encoder pretrained in imagenet")
    # from models.erfnet import ERFNet as ERFNet_imagenet
    # pretrainedEnc = ERFNet_imagenet(1000)
    # checkpnt = torch.load('save/erfnet_encoder_pretrained.pth')
    # pretrainedEnc.load_state_dict(checkpnt['state_dict'])
    # pretrainedEnc = next(pretrainedEnc.children()).features.encoder
    if args.model.lower() == 'erfnet':
        print("Model Name: ERFnet")
        model = ERFNet(num_classes)
        train_params = model.parameters()
    else:
        print("Model Name: DeeplabV3+")
        model = DeepLab(num_classes=num_classes) 
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.learning_rate},
                    {'params': model.get_10x_lr_params(), 'lr': args.learning_rate * 10}]

    # Define Optimizer
    if args.optimizer == 'SGD':
        print('Optimizer: SGD')
        optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=args.weight_decay)
    else:
        print('Optimizer: Adam')
        optimizer = optim.Adam(train_params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    print('Base Learning Rate: ',args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None

    metric = IoU(num_classes, ignore_index=ignore_index)

    model = model.cuda()
    criterion = criterion.cuda()
    # Learning rate decay scheduler
    # cosine decay, linear ramp up to higher LR
    # lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)
    lr_updater = Cust_LR_Scheduler(mode='poly', base_lr=args.learning_rate, num_epochs=args.epochs,iters_per_epoch=len(train_loader), lr_step=0,warmup_epochs=1)
    #resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou, val_miou, train_miou, val_loss, train_loss, lr_list = utils.load_checkpoint( model,optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} | Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0
        val_miou = []
        train_miou = []
        val_loss = []
        train_loss = []
        lr_list = []

    # Start Training
    train = Train(model, train_loader, optimizer, criterion, metric,lr_updater)
    val = Test(model, val_loader, criterion, metric)
    vloss = 0.0
    miou = 0.0   
    for epoch in range(start_epoch, args.epochs):
        print(">> [Epoch: {0:d}] Training LR: {1:.8f}".format(epoch,lr_updater.get_LR(epoch)))
        # lr_updater.step()
        epoch_loss, (iou, tmiou) = train.run_epoch(epoch)
        if vloss == 0.0:
            vloss = epoch_loss
        print(">> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, epoch_loss, tmiou))
        #preform a validation test
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))
            vloss, (iou, miou) = val.run_epoch()
            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, vloss, miou))
            # Save the model if it's the best thus far
            if miou > best_miou:
                print("Best model thus far. Saving...")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou, val_miou, train_miou, val_loss, train_loss, lr_list, args)

        train_loss.append(epoch_loss)
        train_miou.append(tmiou)
        val_loss.append(vloss)
        val_miou.append(miou)
        lr_list.append(lr_updater.get_LR(epoch))

    return model, train_loss, train_miou, val_loss, val_miou,lr_list

def test(model, test_loader, class_weights, class_encoding):
    print("Testing...")
    num_classes = len(class_encoding)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if use_cuda:
        criterion = criterion.cuda()

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, use_cuda)

    print(">>>> Running test dataset")
    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))
    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

def single():
    print('Mode: Single')
    img = Image.open('test_content/dusseldorf_000000_000019_leftImg8bit.png').convert('RGB')
    class_encoding = color_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),               #0
            ('road', (128, 64, 128)),               #1
            ('sidewalk', (244, 35, 232)),           #2
            ('building', (70, 70, 70)),             #3
            ('wall', (102, 102, 156)),              #4
            ('fence', (190, 153, 153)),
            ('pole', (153, 153, 153)),
            ('traffic_light', (250, 170, 30)),
            ('traffic_sign', (220, 220, 0)),
            ('vegetation', (107, 142, 35)),
            ('terrain', (152, 251, 152)),           #10
            ('sky', (70, 130, 180)),
            ('person', (220, 20, 60)),
            ('rider', (255, 0, 0)),
            ('car', (0, 0, 142)),                   #14
            ('truck', (0, 0, 70)),
            ('bus', (0, 60, 100)),
            ('train', (0, 80, 100)),
            ('motorcycle', (0, 0, 230)),
            ('bicycle', (119, 11, 32)) ])
    # label_to_rgb = transforms.Compose([utils.LongTensorToRGBPIL(class_encoding),transforms.ToTensor()])
    # label = utils.PILToLongTensor()(label)
    # color_predictions = utils.batch_transform(label, label_to_rgb)
    # plt.subplot(2,1,1)
    # plt.imshow(img)
    # plt.subplot(2,1,2)
    # plt.imshow(color_predictions)
    
    num_classes = len(class_encoding)
    if args.model.lower() == 'erfnet':
        print("Model Arch: ERFnet")
        model = ERFNet(num_classes)
    else:
        print("Model Arch: DeeplabV3+")
        model = DeepLab(backbone='resnet', output_stride=8, num_classes=20,freeze_bn=True)

    print('model name: ',args.name)
    model_path = os.path.join(args.save_dir, args.name)
    print('Loading model at:',model_path)
    checkpoint = torch.load(model_path)
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    img = img.resize((args.width, args.height), Image.BICUBIC)
    start = time.time()
    images = transforms.ToTensor()(img)
    torch.reshape(images, (1, 3, args.width, args.height))
    images= images.unsqueeze(0)
    with torch.no_grad():
        images = images.cuda()
        predictions = model(images) 
        end = time.time()
        print('model speed:',int(1/(end - start)),"FPS")
        _, predictions = torch.max(predictions.data, 1)
        label_to_rgb = transforms.Compose([utils.LongTensorToRGBPIL(class_encoding),transforms.ToTensor()])
        color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
        end = time.time()
        print('model+transform:',int(1/(end - start)),"FPS")
        utils.imshow_batch(images.data.cpu(), color_predictions)

if __name__ == '__main__':
    if args.mode.lower() == 'single':
        single()
    elif args.mode.lower() == 'vidpred':
        vidpred(args)
    elif args.mode.lower() == 'frames2vid':
        frames2vid(args)
    else:
        if args.dataset.lower() == 'camvid':
            from data import CamVid as dataset
        elif args.dataset.lower() == 'cityscapes':
            from data import Cityscapes as dataset
        elif args.dataset.lower() == 'kitti':
            from data import Kitti as dataset
        elif args.dataset.lower() == 'imagenet':
            from data import Imagenet as dataset
        else:
            raise RuntimeError("\"{0}\" is not a supported dataset.".format(args.dataset))
        
        loaders, w_class, class_encoding = load_dataset(dataset)
        train_loader, val_loader, test_loader = loaders
        
        if args.mode.lower() == 'train':
            from matplotlib.pyplot import figure
            figure(num=None, figsize=(16, 12), dpi=250, facecolor='w', edgecolor='k')
            model,tl,tmiou,vl,vmiou,lrlist = train(train_loader, val_loader, w_class, class_encoding)
            numepc = len(tl)
            plt.plot(tl,label="train loss")
            plt.plot(vl,label="val loss")
            plt.legend() 
            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("loss")
            plt.xticks(np.arange(0, numepc, step=20))
            plt.savefig('./plots/loss.png')
            plt.clf()
            plt.plot(tmiou,label="train miou")
            plt.plot(vmiou,label="val miou")
            plt.legend() 
            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("class miou")
            plt.xticks(np.arange(0, numepc, step=20))
            plt.savefig('./plots/miou.png')
            plt.clf()
            plt.plot(lrlist,label='LR')
            plt.legend() 
            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.xticks(np.arange(0, numepc, step=20))
            plt.savefig('./plots/LR.png')

        elif args.mode.lower() == 'test':
            num_classes = len(class_encoding)
            if args.model.lower() == 'erfnet':
                model = ERFNet(num_classes)
            else:
                model = DeepLab(num_classes)
            model = model.cuda()
            optimizer = optim.Adam(model.parameters())
            model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]
            test(model, test_loader, w_class, class_encoding)
        else:
            raise RuntimeError("\"{0}\" is not a valid choice for execution mode.".format(args.mode))
