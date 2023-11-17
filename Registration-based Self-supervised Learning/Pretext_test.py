# python imports
import copy
import os
import glob
# external imports
import random
import shutil
import cv2

import torch
import numpy as np
from torch.utils.data import DataLoader
from data_set import *
from model import SUNet, SpatialTransformer



def seed_torch(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model_path = 'epoch_100.pth'
    batch_size = len(train_dataset)
    test_dataset = DataLoader(train_dataset, batch_size=1, num_workers=0)

    net = SUNet(2, 2).to(device)

    STN = SpatialTransformer((96, 144)).to(device)
    seed_torch()
    if torch.cuda.is_available():
        net.cuda()
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    net.eval()

    with torch.no_grad():

        for data in test_dataset:


            input_fixed, input_moving, label, filenames = data

            for i in range(len(input_fixed)):

                test_input_moving = input_moving[i].unsqueeze(0).to(device).float()
                test_input_fixed = input_fixed[i].unsqueeze(0).to(device).float()
                label = label[i].unsqueeze(0).to(device).float()
                flow_m2f = net(test_input_moving, test_input_fixed)
                m2f = STN(test_input_moving, flow_m2f.to(device))
                filename =filenames[i]

                save_to_probability_graph(test_input_fixed, test_input_moving,  m2f, label, filename)

def save_to_probability_graph(images,tran_images, reg_image, label, filename):
            path = './PH/'



            temp_string = filename[3:-7]
            temp_string = temp_string.replace('/', '_')


            dir_path = path + temp_string
            os.makedirs(dir_path)
            images = exchange(images)
            tran_images = exchange(tran_images)
            reg_image = exchange(reg_image)
            label = exchange(label)


            cv2.imwrite(dir_path + '/original.jpg', images)
            cv2.imwrite(dir_path + '/warped_image.jpg', tran_images)
            cv2.imwrite(dir_path + '/reg_image.jpg', reg_image)
            cv2.imwrite(dir_path + '/label.jpg', label)


def exchange(input_tensor):


                    input_tensor = input_tensor.clone().detach()
                    input_tensor = input_tensor.to(torch.device('cpu'))
                    input_tensor = input_tensor.squeeze()
                    input_tensor = input_tensor.mul_(255).type(
                        torch.uint8).numpy()
                    return input_tensor






if __name__ == "__main__":
   main()

