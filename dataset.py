import torch.utils.data as Data
from PIL import Image
import cv2
from imageio import imread
import numpy as np
import os
import config
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import math
import random
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur
from ag import IsotropicResize

class Dataset_seg_baseline:
    def __init__(self, datatype, datapath):
        self.datatype = datatype
        self.datapath = datapath
        fh = open(config.DATA_PATH + datapath +'/'+ datatype + '.txt', 'r')
        data_content = fh.read().splitlines()
        imgs = list([])
        for img_line in data_content:
            words = img_line.rstrip().split(' ')
            imgs.append((int(words[0]), words[1]))
        self.imgs = imgs
        size = 299
        
        self.transform =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.XCEPTION['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(*config.XCEPTION['norms'])
        ])
        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.XCEPTION['map_size']),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def argu(self, image, mask, generation=False):
        size = 299
        if generation:
            return Compose([ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                GaussNoise(p=0.1),
                GaussianBlur(blur_limit=3, p=0.05),
                HorizontalFlip(),
                OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ], p=1),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                ToGray(p=0.2),])(image=image, mask=mask)
        else:
            return Compose([ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                GaussNoise(p=0.1),
                GaussianBlur(blur_limit=3, p=0.05),
                HorizontalFlip(),
                ToGray(p=0.2),])(image=image, mask=mask)
            
    def load_image(self, path):
        try:
            img = cv2.imread(path)
        except Error:
            print('no img', path)
        return img

    def load_gt_mask(self, path):
        try:
            img = cv2.imread(path)
        except Error:
            print('no mask', path)
        return img
      
    def __getitem__(self, index):
        img_item = self.imgs[index]
        img = self.load_image(config.DATA_SOURCE_PATH + img_item[1])
        label = img_item[0]
        if 'train' in self.datatype :
            gt_path = img_item[1].replace('ffpp', 'ffpp_seg_ssim').replace('c23', 'raw').replace('c40', 'raw')
            gt_msk = self.load_gt_mask(config.DATA_SOURCE_PATH + gt_path)
            if random.random() > 0.5:
                image = self.argu(img, gt_msk)
                img = image['image']
                gt_msk = image['mask']
            return self.transform(img),self.transform_mask(gt_msk), label
       
        return self.transform(img), label
        
    def __len__(self):
        return len(self.imgs)

class Dataset_srm:
    def __init__(self, datatype, datapath):
        self.datatype = datatype
        self.datapath = datapath
        fh = open(config.DATA_PATH + datapath +'/'+ datatype + '.txt', 'r')
        data_content = fh.read().splitlines()
        imgs = list([])
        for img_line in data_content:
            words = img_line.rstrip().split(' ')
            imgs.append((int(words[0]), words[1]))
        self.imgs = imgs
        size = 299

        self.transform =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.XCEPTION['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(*config.XCEPTION['norms'])
        ])



    def argu(self, image, generation=False):
        size = 299
        if generation:
            return Compose([ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                GaussNoise(p=0.1),
                GaussianBlur(blur_limit=3, p=0.05),
                HorizontalFlip(),
                OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ], p=1),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                ToGray(p=0.2),])(image=image)
        else:
            return Compose([ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                GaussNoise(p=0.1),
                GaussianBlur(blur_limit=3, p=0.05),
                HorizontalFlip(),
                ToGray(p=0.2),])(image=image)
            
    def load_image(self, path):
        try:
            img = cv2.imread(path)
        except Error:
            print('no img', path)
        return img

    def load_gt_mask(self, path):
        try:
            img = cv2.imread(path)
        except Error:
            print('no mask', path)
        return img
      
    def __getitem__(self, index):
        img_item = self.imgs[index]
        img = self.load_image(config.DATA_SOURCE_PATH + img_item[1])
        label = img_item[0]
        if 'train' in self.datatype  and random.random() > 0.5:
                image = self.argu(img)
                img = image['image']
        img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.imgs)

class Dataset_srm_seg:
    def __init__(self, datatype, datapath):
        self.datatype = datatype
        self.datapath = datapath
        fh = open(config.DATA_PATH + datapath +'/'+ datatype + '.txt', 'r')
        data_content = fh.read().splitlines()
        imgs = list([])
        for img_line in data_content:
            words = img_line.rstrip().split(' ')
            imgs.append((int(words[0]), words[1]))
        self.imgs = imgs
        size = 299

        self.transform =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.XCEPTION['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(*config.XCEPTION['norms'])
        ])

        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.XCEPTION['map_size']),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])


    def argu(self, image,mask, generation=True):
        size = 299
        if generation:
            return Compose([ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                GaussNoise(p=0.1),
                GaussianBlur(blur_limit=3, p=0.05),
                HorizontalFlip(),
                OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ], p=1),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                ToGray(p=0.2),])(image=image, mask=mask)
        else:
            return Compose([ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                GaussNoise(p=0.1),
                GaussianBlur(blur_limit=3, p=0.05),
                HorizontalFlip(),
                ToGray(p=0.2),])(image=image, mask=mask)
            
    def load_image(self, path):
        try:
            img = cv2.imread(path)
        except Error:
            print('no img', path)
        return img

    def load_gt_mask(self, path):
        try:
            img = cv2.imread(path)
        except Error:
            print('no mask', path)
        return img
      
    def __getitem__(self, index):
        img_item = self.imgs[index]
        img = self.load_image(config.DATA_SOURCE_PATH + img_item[1])
        
        label = img_item[0]
        if 'train' in self.datatype:
            gt_path = img_item[1].replace('ffpp', 'ffpp_seg_ssim').replace('c23', 'raw').replace('c40', 'raw')
            gt_msk = self.load_gt_mask(config.DATA_SOURCE_PATH + gt_path)
            if random.random() > 0.5:
                image = self.argu(img, gt_msk)
                img = image['image']
                gt_msk = image['mask']
            return self.transform(img),self.transform_mask(gt_msk), label
                
        return self.transform(img), label
        
    def __len__(self):
        return len(self.imgs)

class Dataset_baseline:
    def __init__(self, datatype, datapath):
        self.datatype = datatype
        self.datapath = datapath

        fh = open(config.DATA_PATH + datapath +'/'+ datatype + '.txt', 'r')
        data_content = fh.read().splitlines()
        #print(len(content))
        imgs = list([])
        for img_line in data_content:
            words = img_line.rstrip().split(' ')
            imgs.append((int(words[0]), words[1]))
        self.imgs = imgs

        self.transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.XCEPTION['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(*config.XCEPTION['norms'])
        ])
        size = 299
        
        self.transform =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.XCEPTION['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(*config.XCEPTION['norms'])
        ])
        
    def argu(self, image):
        size = 299
        return Compose([ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),])(image=image)

    def load_image(self, path):
        img = cv2.imread(path)
        if self.datatype == 'train' and random.random() > 0.5:
            img = cv2.resize(img, (299,299))
            img = self.argu(img)
            # print(img)
            img = self.transform(img['image'])
            return img
        return self.transform(img)

    def __getitem__(self, index):
        img_item = self.imgs[index]
        # print('img', config.DATA_SOURCE_PATH + img_item[1])
        img = self.load_image(config.DATA_SOURCE_PATH + img_item[1])
        label = img_item[0]
        if self.datatype == 'eval' and (self.datapath == 'ffpp_c0' or self.datapath == 'ffpp_c23' or self.datapath == 'ffpp_c40'):
            a = img_item[1].replace('ffpp/', '').replace('manipulated/','').split('/')[0]
            # print(a)
            # print(self.type_index[a])
            return img, label, self.type_index[a]
        return img, label

    def __len__(self):
        return len(self.imgs)




def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in tqdm(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())
 
if __name__ == '__main__':
    train_dataset = Dataset_patch_dct_no_mask('eval','ffpp_c23')
    print(getStat(train_dataset))
