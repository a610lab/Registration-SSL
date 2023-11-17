import torch
from torch.utils.data import Dataset
from PIL import Image
import nibabel as nib
import numpy as np
import cv2
import image_utils


class MyDataSet(Dataset):
    """ make_dataset """

    def __init__(self, images_path: list, labels_path: list, image_size=(96, 144), aug=True):
        self.images_path = images_path
        self.labels_path = labels_path
        self.image_size= image_size
        self.aug = aug

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        nib_image = nib.load(self.images_path[item])
        nib_label = nib.load(self.labels_path[item])
        whole_image = np.array(nib_image.get_data(), dtype='float32')
        whole_label = np.array(nib_label.get_data(), dtype='float32')
        resolution = nib_image.header['pixdim'][1:3]
        filename = self.images_path[item]

        return whole_image, whole_label, resolution, filename

    @staticmethod
    def collate_fn(batch):
        images, labels, resolutions, filenames = tuple(zip(*batch))

        # tensor_images = torch.tensor(images)
        # tensor_labels = torch.tensor(labels)
        # tensor_images=tensor_images.cuda()
        # tensor_labels=tensor_labels.cuda()
        # return tensor_images, tensor_labels
        return images, labels, resolutions, filenames


