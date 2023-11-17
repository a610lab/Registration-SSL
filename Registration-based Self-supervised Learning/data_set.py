import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
from torchvision import transforms as transforms
from transform import *
class MyDataSet(Dataset):
    """ make_dataset """

    def __init__(self, images_path: list, labels_path: list, image_size=(96, 144), transform = None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        nib_image = nib.load(self.images_path[item])
        nib_label = nib.load(self.labels_path[item])
        imgs = nib_image.get_data()
        img = cv2.resize(imgs, (144, 96))
        labels = nib_label.get_data().astype(np.uint8)
        label = cv2.resize(labels, (144, 96))
        filename = self.images_path[item]
        path = 'zb.xlsx'
        data = pd.read_excel(path)
        datas = data.values
        name = filename
        for i in range(len(datas)):
            if name == datas[i][0]:
                a = np.array(eval(datas[i][1]))
                c = []
                for number in a:
                    x, y = number
                    c.append(generate_points(x, y))
                b = c
                def tran(img, pi=a, qi=b):
                    ddd = trans(img, pi)
                    img = ddd.deformation(img, qi)
                    return img
                t_img = tran(img)
                t_img = self.data_preprocess(t_img)
                t_img = transform(t_img)
        img = self.data_preprocess(img)
        img = transform(img)
        label =self.data_preprocess(label)
        label = transform(label)
        filename = self.images_path[item]


        return img, t_img, label, filename

    def data_preprocess(self, image):

        clip_min = np.percentile(image, 1)
        clip_max = np.percentile(image, 99)
        image = np.clip(image, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        return image



def generate_points(center_x, center_y):
                    radius2 = np.random.randint(-5, 5)
                    radius1 = np.random.randint(-3, 3)
                    x = center_x + radius1
                    y = center_y + radius2
                    points = (x, y)
                    return points

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


rootpath = 'E:/Data'
train_listfile = 'E:/dj/Data/sparc_data_split/filelist_train99_r0'
train_listfile1 = 'E:/Data/sparc_data_split/filelist_test44'
train_images_paths, train_labels_paths= read_datalist(rootpath, train_listfile)
train_images_paths1, train_labels_paths1= read_datalist(rootpath, train_listfile1)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MyDataSet(images_path=train_images_paths, labels_path=train_labels_paths, transform = transform)
train_dataset2 = MyDataSet(images_path=train_images_paths1, labels_path=train_labels_paths1, transform = transform)
train_dataset = train_dataset + train_dataset2




