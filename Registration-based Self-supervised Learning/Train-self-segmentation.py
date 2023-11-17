import torch
# 加载优化器模块
import torch.optim as optim
# 加载数据集模块
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn.functional as F
# 加载图像转换类
# 加载操作系统相关的调用和操作
import os
# JSON是一种轻量级的数据交换格式，易于人阅读和编写
from tensorboardX import SummaryWriter
from tqdm import tqdm
from model import SUNet
# from network import UNet
from LoadData import MyDataSet, transform
import numpy as np
import random
import csv


def seed_torch(seed):
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
    seed_torch(seed=fixed_seed)
    # 设置批量
    batch_size = 16
    learning_rate = 0.0001
    epochs = 50
    rootpath = '../Data'


    for index in range(K_fold):
        note = "{}_SDU_K{}-{}_2".format(fixed_seed, K_fold, index+1)
        train_listfile = '../Data/sparc_data_split/filelist_train99_r0'

        fid_csv = open( './train-file/ '+ '{}.csv'.format(note), 'w', encoding='utf-8')
        writer = SummaryWriter('./graph/' + note)

        csv_writer = csv.writer(fid_csv)
        csv_writer.writerow(["parameters", "batch_size", "learning_rate", "epochs", "path_pretask_model"])
        csv_writer.writerow([batch_size, learning_rate, epochs])
        csv_writer.writerow(["epoch", "train_loss", "train_dsc", "valid_loss", "valid_dsc"])
        train_images_paths, train_labels_paths = read_datalist(rootpath, train_listfile)


        print('\033[0;31;40mStart Training {}\033[0m'.format(note))

        path = './weights/'
        save_path = path + note + '.pth'

        ###===================Load Data============================

        if K_fold == 1:
            percnt = 1
        elif K_fold == 2:
            percnt = 1 / 2
        elif K_fold == 3:
            percnt = 1 / 3
        else:
            percnt = 1 / 10


        random.seed(fixed_seed)
        random.shuffle(train_images_paths)
        random.seed(fixed_seed)
        random.shuffle(train_labels_paths)

        begin = int(len(train_images_paths) * percnt * index)
        end = int(len(train_images_paths) * percnt * (index + 1))
        subtrain_images_paths = train_images_paths[begin:end]
        subtrain_labels_paths = train_labels_paths[begin:end]


        train_dataset = MyDataSet(images_path=subtrain_images_paths, labels_path = subtrain_labels_paths, transform=transform)
        train_num = len(train_dataset)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                 )




        print("using {} images for training." .format(train_num))


        ###===================Segmentation Task Training============================
        net =SUNet(1, 1)
        if torch.cuda.is_available():
            net.cuda()



        path = 'epoch_100.pth'
        save_model = torch.load(path)
        model_dict = net.state_dict()
        save_model.pop("U_net1.in_conv.0.weight")
        save_model.pop("U_net1.out_conv.0.weight")
        save_model.pop("U_net2.out_conv.0.bias")
        save_model.pop("U_net2.in_conv.0.weight")
        save_model.pop("U_net1.out_conv.0.bias")
        save_model.pop("U_net2.out_conv.0.weight")
        model_dict.update(save_model)
        net.load_state_dict(model_dict)

        net.to(device)


        optimizer = optim.Adam(net.parameters(), lr=learning_rate)


        final_train_dsc = 0.0
        best_train_dsc = 0.0

        for epoch in range(epochs):

            # train
            net.train()
            train_bar = tqdm(train_loader)

            avdsc_train = 0.0
            avloss_train = 0.0
            for step, data in enumerate(train_bar):
                images, labels = data
                logits = net(images.to(device))

                labels = labels.to(device)
                loss = F.binary_cross_entropy_with_logits(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avloss_train += loss
                dsc = DSC(logits, labels.to(device))

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f} dsc:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss, dsc)
                avdsc_train += dsc

            avdsc_train /= len(train_bar)
            avloss_train /= len(train_bar)



            print("Training average DSC:{:.3f}".format(avdsc_train))


            if avdsc_train > best_train_dsc:
                best_train_dsc = avdsc_train
                final_train_dsc = avdsc_train
                torch.save(net.state_dict(), save_path)

            writer.add_scalar("{}Train/Loss".format(note), avloss_train, epoch)
            writer.add_scalar("{}Train/DSC".format(note), avdsc_train, epoch)

            csv_writer.writerow(
                [index, epoch, avloss_train.item(), avdsc_train.item()])
            writer.close()

        csv_writer.writerow(
            ["{} results:".format(note),
             "final_train_dsc", final_train_dsc.item(),
             "best_train_dsc", best_train_dsc.item(),
            ])

    fid_csv.close()
    print('Finished Training')


def DSC(input: Tensor, target: Tensor):
    input = F.sigmoid(input)
    input[input >= 0.5] = 1
    input[input < 0.5] = 0
    input_flatten = input.flatten()
    target_flatten = target.flatten()
    intersection = torch.sum(input_flatten * target_flatten)
    dsc = (2. * intersection) / (torch.sum(input_flatten) + torch.sum(target_flatten))

    return dsc

def read_datalist(rootpath, listfile):
    train_images_paths = list()
    train_labels_paths = list()

    file = open(listfile)

    for line in file:
        line = line.strip('\n')
        tmpimagefile = rootpath + line[1:]
        tmplabelfile = rootpath + line[1:-7] + '_label.nii.gz'

        train_images_paths.append(tmpimagefile)
        train_labels_paths.append(tmplabelfile)
    return train_images_paths, train_labels_paths



if __name__ == '__main__':
    seeds = [19]
    K_folds = [10]
    for s in seeds :
        for k in K_folds:
            fixed_seed = s
            K_fold = k
            main()



