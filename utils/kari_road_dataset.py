import torch
from pathlib import Path
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class KariRoadDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=False):
        self.root = Path(root)
        self.train = train
        if train:
            self.img_dir = self.root/'train'/'images'
        else:
            self.img_dir = self.root/'val'/'images'
        self.img_files = sorted(self.img_dir.glob('*.png'))
        self.transform = get_transforms(train)

    def __getitem__(self, idx):
        img_file= self.img_files[idx].as_posix()
        label_file = img_file.replace('images', 'labels')
        img = cv2.imread(img_file)
        label_img = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        img, label_img = self.transform(img, label_img)
        return img, label_img, img_file

    def __len__(self):
        return len(self.img_files)
    
class ImageAug:
    def __init__(self, train):
        if train:
            self.aug = A.Compose([A.RandomCrop(256, 256),
                                  A.HorizontalFlip(p=0.5),
                                  A.ShiftScaleRotate(p=0.3),
                                  A.RandomBrightnessContrast(p=0.3),
                                  A.pytorch.transforms.ToTensorV2()])
        else:
            self.aug = ToTensorV2()

    def __call__(self, img, label_img):
        transformed = self.aug(image=img, mask=np.squeeze(label_img))
        return transformed['image']/255.0, transformed['mask']

def get_transforms(train):
    transforms = ImageAug(train)
    return transforms