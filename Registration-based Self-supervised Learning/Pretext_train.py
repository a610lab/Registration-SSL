import os
from random import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import SUNet, SpatialTransformer
import  losses
from data_set import *
import torch


seed=10
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    net = SUNet(2, 2).to(device)
    STN = SpatialTransformer((96, 144)).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    DL = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
    epoch = 200
    total_train_step = 0
    best_loss = 0.0
    writer = SummaryWriter("../logs_trains12")
    # Training loop.
    for i in range(epoch):
        net.train()

        train_bar = tqdm(DL)
        # Generate the moving images and convert them to tensors.

        avloss_train = 0.0
        for step, data in enumerate(train_bar):
            input_fixed, input_moving = data
            print(input_moving.shape)
            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()
            flow_m2f = net(input_moving, input_fixed)
            print(flow_m2f.shape)
            m2f = STN(input_moving, flow_m2f.to(device))
            sim_loss = losses.NCC().loss(m2f.to(device), input_fixed)
            loss = sim_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avloss_train +=loss
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(i + 1,epoch,loss, )
            total_train_step = total_train_step + 1
            if total_train_step % 10 == 0:
                writer.add_scalar("train_loss", loss.item(), total_train_step)
        avloss_train = avloss_train / len(DL)
        print("Training average loss:{:.3f}" .format(avloss_train))
        if (i+1) % 10 == 0:
            save_file_name = 'Zn_pretask{}.pth'.format(i+1)
            torch.save(net.state_dict(), save_file_name)
    writer.close()

if __name__ == "__main__":
   main()


