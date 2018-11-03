import argparse
import os
import cv2 as cv 

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
from utils import plot_img_and_mask

from torchvision import transforms

def PreProcess(img_path, resize_shape=512):
    img = cv.imread(img_path)            
    resized_img = cv.resize(img, (resize_shape, resize_shape),
                            interpolation = cv.INTER_NEAREST)
    resized_img = np.transpose(resized_img, (2, 0, 1)) / 255
    resized_img = np.expand_dims(resized_img, axis=0)
    return resized_img.astype(np.float32)

def predict_img(net,
                img_path,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):
    net.eval()
    img = torch.from_numpy(PreProcess(img_path))
    mask_softmax = torch.softmax(net(img), dim=1)
    #mask_mul = torch.ge(torch.max(mask_softmax, dim=1)[0], 0.5).int()
    mask_out = torch.argmax(mask_softmax, dim=1)
    return mask_out.numpy().astype(np.uint8)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--num_cls', default=21, type=int, metavar='num_cls',
                        help='number of classes includes background')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no_save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no_crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask_threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 10).astype(np.uint8))

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.num_cls)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        mask = predict_img(net=net,
                           img_path=fn,
                           out_threshold=args.mask_threshold,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)
        _, h, w = mask.shape
        mask = mask.reshape((h, w, 1))
        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            img = cv.imread(fn)
            cv.imshow('sadfasdf', img)
            cv.imshow('dasfdasfdasf', mask)  
            cv.waitKey()         

        if not args.no_save:
            out_fn = out_files[i]
            cv.imwrite(out_files[i], mask)

            print("Mask saved to {}".format(out_files[i]))
