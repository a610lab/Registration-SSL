
import numpy as np
import cv2
import sys
# from itertools import combinations, permutations
import random

def data_preprocess(image, label, image_size, augment=False):
    clip_min = np.percentile(image, 1)
    clip_max = np.percentile(image, 99)
    image = np.clip(image, clip_min, clip_max)
    image = (image - image.min()) / float(image.max() - image.min())

    resized_image = cv2.resize(image, (image_size[1], image_size[0]), cv2.INTER_CUBIC)
    resized_label = cv2.resize(label, (image_size[1], image_size[0]), cv2.INTER_NEAREST)

    # Perform data augmentation
    if augment:
        resized_image, resized_label = augment_data(resized_image, resized_label, shift=20, rotate=30,
                                                                scale=0.2, intensity=0.3, flip=True)

    resized_image = np.expand_dims(resized_image, axis=0)
    resized_label = np.expand_dims(resized_label, axis=0)
    return resized_image, resized_label



def crop_image(image, crop_shape, cropmode='center', left_top_point=(0,0)):

    cropped_image = np.zeros((crop_shape[0], crop_shape[1], image.shape[-1]), image.dtype)

    if cropmode == 'center':

        if (image.shape[0] < crop_shape[0]) or (image.shape[1] < crop_shape[1]):
            pady = np.abs(crop_shape[0] - image.shape[0])
            padx = np.abs(crop_shape[1] - image.shape[1])

            padded_image = np.pad(image, ((int(pady/2), pady-int(pady/2)), (int(padx/2), padx-int(padx/2)), (0, 0)),
                                  'constant', constant_values=(0, 0))
        else:
            padded_image = image

        topy = int((padded_image.shape[0] - crop_shape[0]) / 2)
        bottomy = topy + crop_shape[0]
        leftx = int((padded_image.shape[1] - crop_shape[1]) / 2)
        rightx = leftx + crop_shape[1]
        cropped_image = padded_image[topy:bottomy, leftx:rightx, :]

    if cropmode == 'handcraft':
        assert (left_top_point[0] + crop_shape[0]) <= image.shape[0]
        topy = left_top_point[0]
        bottomy = left_top_point[0] + crop_shape[0]
        assert (left_top_point[1] + crop_shape[1]) <= image.shape[1]
        leftx = left_top_point[1]
        rightx = left_top_point[1] + crop_shape[1]
        cropped_image = image[topy:bottomy, leftx:rightx, :]
    if (cropmode != 'center') and (cropmode != 'handcraft'):
        ValueError("Error cropping mode!")
        sys.exit(0)

    return cropped_image


def pad_image(image, pad_shape, pad_mode='center'):

    padded_image = np.zeros((pad_shape[0], pad_shape[1], image.shape[-1]), image.dtype)

    if pad_mode == 'center':

        pady = np.abs(pad_shape[0] - image.shape[0])
        padx = np.abs(pad_shape[1] - image.shape[1])

        padded_image = np.pad(image, ((int(pady/2), pady-int(pady/2)), (int(padx/2), padx-int(padx/2)), (0, 0)),
                              'constant', constant_values=(0, 0))

        topy = int((padded_image.shape[0] - pad_shape[0]) / 2)
        bottomy = topy + pad_shape[0]
        leftx = int((padded_image.shape[1] - pad_shape[1]) / 2)
        rightx = leftx + pad_shape[1]
        padded_image = padded_image[topy:bottomy, leftx:rightx, :]

    if (pad_mode != 'center'):
        ValueError("Error cropping mode!")
        sys.exit(0)

    return padded_image


def pad_image2d(image, pad_shape, pad_mode='center'):

    if pad_mode == 'center':

        pady = np.abs(pad_shape[0] - image.shape[0])
        padx = np.abs(pad_shape[1] - image.shape[1])

        padded_image = np.pad(image, ((int(pady/2), pady-int(pady/2)), (int(padx/2), padx-int(padx/2))),
                              'constant', constant_values=(0, 0))

        topy = int((padded_image.shape[0] - pad_shape[0]) / 2)
        bottomy = topy + pad_shape[0]
        leftx = int((padded_image.shape[1] - pad_shape[1]) / 2)
        rightx = leftx + pad_shape[1]
        padded_image = padded_image[topy:bottomy, leftx:rightx]

    if (pad_mode != 'center'):
        ValueError("Error cropping mode!")
        sys.exit(0)

    return padded_image

def random1():

    return 2 * (np.random.rand(1) - 0.5)



def augment_data(image, label, shift=30, rotate=60, scale=0.2, intensity=0.2, flip=False):
    aug_image = np.ndarray(image.shape, image.dtype)
    aug_label = np.ndarray(label.shape, label.dtype)
    h, w = image.shape
    x_centre, y_centre = int(w / 2), int(h / 2)


    shiftx = int(shift * random1())
    shifty = int(shift * random1())

    angle = float(rotate * random1())

    scale_ratio = float(1 + scale * random1())

    image = image * (1 + intensity * random1())

    rot_mat = cv2.getRotationMatrix2D((x_centre + shiftx, y_centre + shifty), angle, scale_ratio)

    image_slice = image[:, :]
    label_slice = label[:, :]

    aug_image_slice = cv2.warpAffine(image_slice, rot_mat, (image_slice.shape[1], image_slice.shape[0]))
    aug_label_slice = cv2.warpAffine(label_slice, rot_mat, (label_slice.shape[1], label_slice.shape[0]), flags=cv2.INTER_NEAREST)


    if flip:
        para_flip = (-1, 0, 1, 2)
        ind_flid = random.randint(0,3)
        aug_image_slice = cv2.flip(aug_image_slice, para_flip[ind_flid])
        aug_label_slice = cv2.flip(aug_label_slice, para_flip[ind_flid])



    aug_image[:, :] = aug_image_slice
    aug_label[:, :] = aug_label_slice
    return aug_image, aug_label