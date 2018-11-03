import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from eval import eval_net
from unet import UNet
from dataloader import SegLoader 
from torch.utils.data import Dataset, DataLoader

def train_net(net,
              dir_img,
              dir_mask,
              dir_checkpoint, 
              epochs=5,
              batch_size=1,
              lr=0.1,
              lr_step=10,
              val_percent=0.5,
              save_cp=True,
              gpu=False,
              img_scale=1.0):

    train_data = SegLoader(dir_img = dir_img, dir_label = dir_mask, 
                           resize_shape = 512)
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr,
               str(save_cp), str(gpu)))

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step)
    cls_weight = torch.ones(21)
    cls_weight[0] = 0.1
    criterion = nn.CrossEntropyLoss(weight=cls_weight)
    for epoch in range(epochs):
        scheduler.step()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        epoch_loss = 0
        dataloader = DataLoader(train_data, batch_size=batch_size,
                                shuffle=True, num_workers=4)
        for data in dataloader:
            imgs, true_masks = data['img'], data['label'] 

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
           # masks_probs_flat = masks_probs.view(-1)

           # true_masks_flat = true_masks.view(-1)
            loss = criterion(masks_pred, true_masks.long())
            epoch_loss += loss.item()

            #print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
            print loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print('Epoch finished ! Loss: {}'.format(epoch_loss / r_i))
        """
        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        """
        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('--img_dir', dest='img_dir', type='str',
                      help='path to image directory')
    parser.add_option('--label_dir', dest='label_dir', type='str',
                      help='path to label directory')
    parser.add_option('--save_dir', dest='save_dir', type='str',
                      help='path for saving checkpoints')
    parser.add_option('--num_cls', dest='num_cls', default=21, type='int',
                      help='number of classes includes background')
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning_rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('--lr_step', dest='lr_step', default=10,
                      type='int', help='learning rate multiplied by 0.1 every learning step')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=args.num_cls)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  dir_img = args.img_dir,
                  dir_mask = args.label_dir,
                  dir_checkpoint = args.save_dir,  
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  lr_step=args.lr_step,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
