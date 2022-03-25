import cv2
import numpy as np
import os
from tqdm import tqdm
import mmcv
import os.path as osp
import random


train_ann_dir = './train_val_split/train_labels/'
train_img_dir = './train_val_split/train_images/'

val_ann_dir = './train_val_split/val_labels/'
val_img_dir = './train_val_split/val_images/'

out_dir = "new_tianchi_data/"
if not os.path.exists(os.path.join(out_dir)):
    os.makedirs(os.path.join(out_dir))
if not os.path.exists(os.path.join(out_dir, 'images')):
    os.makedirs(os.path.join(out_dir, 'images'))
if not os.path.exists(os.path.join(out_dir, 'images', 'training')):
    os.makedirs(os.path.join(out_dir, 'images', 'training'))
if not os.path.exists(os.path.join(out_dir, 'images', 'validation')):
    os.makedirs(os.path.join(out_dir, 'images', 'validation'))

if not os.path.exists(os.path.join(out_dir, 'annotations')):
    os.makedirs(os.path.join(out_dir, 'annotations'))
if not os.path.exists(os.path.join(out_dir, 'annotations', 'training')):
    os.makedirs(os.path.join(out_dir, 'annotations', 'training'))
if not os.path.exists(os.path.join(out_dir, 'annotations', 'validation')):
    os.makedirs(os.path.join(out_dir, 'annotations', 'validation'))

# 训练集图片
for img_name in tqdm(os.listdir(train_img_dir)):
    img = mmcv.imread(osp.join(train_img_dir, img_name))

    mmcv.imwrite(
        img,
        osp.join(
            out_dir, 'images', 'training',
            osp.splitext(img_name)[0].replace('_training', '') +
            '.jpg'))

# 验证集图片
for img_name in tqdm(os.listdir(val_img_dir)):
    img = mmcv.imread(osp.join(val_img_dir, img_name))
    mmcv.imwrite(
        img,
        osp.join(
            out_dir, 'images', 'validation',
            osp.splitext(img_name)[0].replace('_training', '') +
            '.jpg'))

for img_name in tqdm(os.listdir(train_ann_dir)):
    cap = cv2.VideoCapture(osp.join(train_ann_dir, img_name))
    ret, img = cap.read()
    # img[:, :, 0] = np.clip(img[:, :, 0], 0, 1)
    # img[:, :, 0] = img[:, :, 0] // 128
    img[:, :, 0] = img[:, :, 0] - 1  # 因为之前的mask处理到了1-2，这里需要-1，处理成0-1
    img_prefix = osp.splitext(img_name)[0] + '.jpg'

    mmcv.imwrite(
        img[:, :, 0],
        osp.join(out_dir, 'annotations', 'training',
                 osp.splitext(img_name)[0] + '.png'))

for img_name in tqdm(os.listdir(val_ann_dir)):
    cap = cv2.VideoCapture(osp.join(val_ann_dir, img_name))
    ret, img = cap.read()
    img[:, :, 0] = img[:, :, 0] - 1  # 因为之前的mask处理到了1-2，这里需要-1，处理成0-1
    img_prefix = osp.splitext(img_name)[0] + '.jpg'
    mmcv.imwrite(
        img[:, :, 0],
        osp.join(out_dir, 'annotations', 'validation',
                 osp.splitext(img_name)[0] + '.png'))

img_train = os.listdir(out_dir + '/images/training')
img_val = os.listdir(out_dir + '/images/validation')
ann_train = os.listdir(out_dir + '/annotations/training')
ann_val = os.listdir(out_dir + '/annotations/validation')
print(len(img_train))
print(len(img_val))
print(len(ann_train))
print(len(ann_val))
