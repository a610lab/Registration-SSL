import copy
import csv
import shutil

import cv2
import torch
# 加载优化器模块
# 加载数据集模块
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn.functional as F
# 加载图像转换类
# 加载操作系统相关的调用和操作
import os
# JSON是一种轻量级的数据交换格式，易于人阅读和编写
import image_utils

import numpy as np
import random
from LoadData_test import MyDataSet
from medpy import metric


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

    seed_torch(seed=fixed_seed)

    K_DSC=[]
    K_DSC_SD=[]
    K_ASSD = []
    K_ASSD_SD = []
    K_HD95 = []
    K_HD95_SD = []
    K_Dlta_PA = []
    K_Dlta_PA_SD = []
    K_Person_r = []


    for index in range(K_fold):

            model_path = './weights/{}_SDU_K{}-{}.pth'.format(fixed_seed, K_fold, index+1)
            note = "{}_SDU_K{}-{}_".format(fixed_seed, K_fold, index+1)


            image_size = (96, 144)

            rootpath = '../Data'

            test_listfile = '../Data/sparc_data_split/filelist_test44'

            fid_csv = open('./test-file/' + '{}.csv'.format(note), 'w', encoding='utf-8')
            csv_writer = csv.writer(fid_csv)
            csv_writer.writerow(
                ["filename", "DSC", "PA_Manual", "PA_Predict", "Delta_PA", "ASSD", "HD95"])


            test_images_paths, test_labels_paths = read_datalist(rootpath, test_listfile)
            test_dataset = MyDataSet(images_path=test_images_paths, labels_path=test_labels_paths)


            ###===================Load Data============================
            test_loader = DataLoader(test_dataset,
                                     batch_size=32,
                                     num_workers=0,
                                     collate_fn=test_dataset.collate_fn
                                     )


            net = SUNet(1, 1)


            if torch.cuda.is_available():
                net.cuda()


            net.load_state_dict(torch.load(model_path, map_location=device))
            net.to(device)

            DSCs = list()
            PAs_pred = list()
            PAs_man = list()
            Delta_PAs = list()
            ASSD = list()
            HD95 = list()
            triple = list()
            num = list()

            net.eval()
            with torch.no_grad():
                for val_data in test_loader:
                    images, labels, resolutions, filenames = val_data
                    result_images = []
                    for ind in range(len(images)):
                        ori_shape = images[ind].shape
                        test_image, test_label = image_utils.data_preprocess(images[ind], labels[ind], image_size, augment=False)
                        test_image = np.expand_dims(test_image, axis=0)
                        test_image = torch.tensor(test_image)
                        outputs = net(test_image.to(device))
                        outputs1 = F.sigmoid(outputs[-1])
                        pred = outputs1.cpu().numpy()
                        resized_pred = cv2.resize(pred[0, :, :], (ori_shape[1], ori_shape[0]), cv2.INTER_NEAREST)
                        result_images.append(copy.deepcopy(resized_pred))

                        resolution = resolutions[ind]
                        filename=filenames[ind]

                        id = filenames[ind][30:36]



                        dsc, pa_pred, pa_man, delta_pa, assd, hd95 = evaluate_metrics_np(resized_pred, labels[ind], resolution)



                        csv_writer.writerow(
                            [filenames[ind], dsc, pa_man, pa_pred, delta_pa, assd, hd95])


                        DSCs.append(dsc)
                        PAs_man.append(pa_man)
                        PAs_pred.append(pa_pred)
                        Delta_PAs.append(delta_pa)
                        ASSD.append(assd)
                        HD95.append(hd95)
                        triple.append((id, pa_man, pa_pred))

            M_TPA = []
            A_TPA = []
            for i in range(len(triple)):
                name = triple[i][0]
                num.append(name)
            person = list(set(num))
            for id in person:
                m_tpa = 0
                a_tpa = 0
                for i in range(len(triple)):
                    if id == triple[i][0]:
                        m_tpa+=triple[i][1]
                        a_tpa+=triple[i][2]
                M_TPA.append(m_tpa)
                A_TPA.append(a_tpa)



                        # save_to_probability_graph(images[ind], result_images[ind], labels[ind], filename)

            avdsc_test = np.mean(DSCs)
            avpa_man_test = np.mean(PAs_man)
            avpa_pred_test = np.mean(PAs_pred)
            avdelta_pa_test = np.mean(Delta_PAs)
            avassd_test = np.mean(ASSD)
            avhd95_test = np.mean(HD95)
            stddsc_test = np.std(DSCs)
            stdpa_man_test = np.std(PAs_man)
            stdpa_pred_test = np.std(PAs_pred)
            stddelta_pa_test = np.std(Delta_PAs)
            stdassd_test = np.std(ASSD)
            stdhd95_test = np.std(HD95)
            person_r = np.corrcoef(M_TPA, A_TPA)[0, 1]


            K_DSC.append(avdsc_test)
            K_DSC_SD.append(stddsc_test)
            K_ASSD.append(avassd_test)
            K_ASSD_SD.append(stdassd_test)
            K_HD95.append(avhd95_test)
            K_HD95_SD.append(stdhd95_test)
            K_Dlta_PA.append(avdelta_pa_test)
            K_Dlta_PA_SD.append(stddelta_pa_test)
            K_Person_r.append(person_r)


            csv_writer.writerow(
                [
                 "avdsc_test", avdsc_test.item(),
                 "avpa_man_test ", avpa_man_test.item(),
                 "avpa_pred_test", avpa_pred_test.item(),
                 "avdelta_pa_test", avdelta_pa_test.item(),
                 "avassd_test", avassd_test.item(),
                 "avhd95_test", avhd95_test.item()
                 ])
            csv_writer.writerow(
                [
                 "stddsc_test", stddsc_test.item(),
                 "stdpa_man_test ", stdpa_man_test.item(),
                 "stdpa_pred_test", stdpa_pred_test.item(),
                 "stddelta_pa_test", stddelta_pa_test.item(),
                 "stdassd_test", stdassd_test.item(),
                 "stdhd95_test", stdhd95_test.item()
                 ])
    av_K_DSC = np.mean(K_DSC)*100
    av_K_DSC_SD = np.mean(K_DSC_SD)*100
    av_K_ASSD = np.mean(K_ASSD)
    av_K_ASSD_SD = np.mean(K_ASSD_SD)
    av_K_HD95 = np.mean(K_HD95)
    av_K_HD95_SD = np.mean(K_HD95_SD)
    av_K_Dlat_PA = np.mean(K_Dlta_PA)
    av_K_Dlta_PA_SD = np.mean(K_Dlta_PA_SD)
    av_K_Persom_r = np.mean(K_Person_r)
    av_r_STD = np.std(K_Person_r)


    # print("DSC:{:.3f},Delta_PA:{:.3f},ASSD:{:.3f},HD95:{:.3f}".format(avdsc_test,  avdelta_pa_test, avassd_test, avhd95_test))
    print("DSC:{:.2f}±{:.2f}, Delta_PA:{:.2f}±{:.2f}, ASSD:{:.2f}±{:.2f}, HD95:{:.2f}±{:.2f}, Person r:{:.3f}±{:.3f}".format(
        av_K_DSC, av_K_DSC_SD, av_K_Dlat_PA, av_K_Dlta_PA_SD, av_K_ASSD, av_K_ASSD_SD, av_K_HD95, av_K_HD95_SD, av_K_Persom_r, av_r_STD))






def evaluate_metrics_np(pred, label, resolution):
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred_flatten = pred.flatten()
        label_flatten = label.flatten()
        intersection = np.sum(pred_flatten * label_flatten)
        dsc = (2. * intersection) / (np.sum(pred_flatten) + np.sum(label_flatten))
        assd = metric.binary.assd(pred, label) * resolution[0]
        hd95 = metric.binary.hd95(pred, label) * resolution[0]
        pa_pred = np.sum(pred_flatten) * resolution[0] * resolution[1]
        pa_lb = np.sum(label_flatten) * resolution[0] * resolution[1]
        delta_pa = abs(pa_pred - pa_lb)
        return dsc, pa_pred, pa_lb, delta_pa, assd, hd95,

def DSC(input: Tensor, target: Tensor):
    input = F.sigmoid(input)
    input[input >= 0.5] = 1
    input[input < 0.5] = 0
    dsc = 0.0
    for ind in range(input.shape[0]):
        input_flatten = input[ind].flatten()
        target_flatten = target[ind].flatten()
        intersection = torch.sum(input_flatten * target_flatten)
        dsc += (2. * intersection) / (torch.sum(input_flatten) + torch.sum(target_flatten))
    dsc /= input.shape[0]
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

def save_to_probability_graph(images, result_images,lanels, filename):
        path = './SP_pre/'
        temp_string = filename[3:-7]
        temp_string = temp_string.replace('/', '_')
        dir_path = path+temp_string
        os.makedirs(dir_path)

        probability_img = result_images * 255
        probability_img = cv2.applyColorMap(probability_img.astype(np.uint8), cv2.COLORMAP_JET)

        mask_img = np.round(result_images) * 255
        label_img = np.round(lanels)*255

        cv2.imwrite(dir_path + '/original.jpg', images)
        cv2.imwrite(dir_path + '/label.jpg', label_img)
        cv2.imwrite(dir_path + '/mask.jpg', mask_img)
        cv2.imwrite(dir_path + '/probability_img.jpg', probability_img)


if __name__ == '__main__':
    fixed_seed = 19
    for k in [10,2]:
        K_fold = k
        main()



