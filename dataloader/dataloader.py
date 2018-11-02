from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import cv2 as cv 
import numpy as np
from PIL import Image

class SegLoader(Dataset):
    def __init__(self, dir_img, dir_label,
                 resize_shape = 500, 
                 img_type=['jpg', 'png']):
        self._img_ids = self.GetImageList(dir_img, img_type)
        self._label_ids = self.GetImageList(dir_label, img_type)  
        assert len(self._img_ids) == len(self._label_ids)
        self._dir_img = dir_img
        self._resize_shape = resize_shape
        self._dir_label = dir_label
             
    def __len__(self):
        return len(self._img_ids)

    def __getitem__(self, idx):
        path_img = os.path.join(self._dir_img, self._img_ids[idx])
        path_label = os.path.join(self._dir_label, self._label_ids[idx])
        img = cv.imread(path_img)
        resized_img = cv.resize(img, (self._resize_shape, self._resize_shape), 
                                interpolation = cv.INTER_NEAREST)
        label = self.RemoveColormap(path_label)  	
        resized_label = cv.resize(label, (self._resize_shape, self._resize_shape),
                                  interpolation = cv.INTER_NEAREST)
        resized_img = np.transpose(resized_img, (2, 0, 1)) / 255
        sample = {'img' : resized_img.astype(np.float32), 'label' : resized_label.astype(np.float32)}
        return sample
 
    def GetImageList(self, dir_path, img_type):
        output = [] 
        for fname in os.listdir(dir_path):
            if fname.split('.')[-1] in img_type:
                output.append(fname) 
        return output

    def Transform(self, img):
        return img

    def RemoveColormap(self, filename):
        img = np.array(Image.open(filename))
        img[img == 255] = 0
        return img

