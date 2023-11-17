import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import nibabel as nib
import numpy as np
import cv2
from torchvision import transforms as transforms
import os


class MyDataSet(Dataset):
    """ make_dataset """

    def __init__(self, images_path: list, labels_path: list, image_size=(96,144), transform = None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        nib_image = nib.load(self.images_path[item])
        nib_label = nib.load(self.labels_path[item])
        # RGB为彩色图片，L为灰度图片

        whole_image = np.array(nib_image.get_data(), dtype='float32')
        whole_label = np.array(nib_label.get_data(), dtype='float32')

        img, label = self.data_preprocess(whole_image, whole_label, self.image_size)
        img = transform(img)
        label = transform(label)

        return img, label

    def data_preprocess(self, image, label, image_size, augment=False):

        clip_min = np.percentile(image, 1)
        clip_max = np.percentile(image, 99)
        image = np.clip(image, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())

        resized_image = cv2.resize(image, (image_size[1], image_size[0]), cv2.INTER_CUBIC)
        resized_label = cv2.resize(label, (image_size[1], image_size[0]), cv2.INTER_NEAREST)

        # Perform data augmentation
        return resized_image, resized_label

transform = transforms.Compose([transforms.ToTensor()])
