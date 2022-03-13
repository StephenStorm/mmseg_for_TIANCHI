import os
import cv2
import os
import random
import numpy as np
import albumentations as albu
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize, Compose
from torch import rand
from tqdm import tqdm

import euler
import thriftpy2.protocol.json as proto
from euler.base_compat_middleware import client_middleware
import time
import os

euler.install_thrift_import_hook()
import sys
sys.path.append('/opt/tiger/workspace/mmsegmentation/mmseg/datasets/pipelines')
from ocrx_demo_forwangsen.idl.ocrx_thrift import OcrService, ImagesOcrRequest, ImageInfo
from ocrx_demo_forwangsen.idl.base_thrift import Base

SERVICE_TARGET = "sd://lab.ocrx.fusion_test?cluster=default&idc=lf"
client = euler.Client(OcrService, SERVICE_TARGET, timeout=30)
client.use(client_middleware)



# 固定随机数种子
# np.random.seed(444)
# random.seed(444)





def get_random_ocr_result(filepath, random_value = 0.5):
    image_binary = ImageInfo(data=open(filepath, "rb").read())
    s_t = time.time()
    req = ImagesOcrRequest(images=[image_binary])
    if "SEC_TOKEN_STRING" in os.environ:
        print("CLL in")
        req.Base = Base(extra={"gdpr-token": os.environ['SEC_TOKEN_STRING']},
                        Caller='lab.ocr.finance')
    #req = ImagesOcrRequest(images=[image_binary], recognition_only=True)
    e_t = time.time()
    # print('time is {}'.format(e_t - s_t))

    ocr_det_res = []
    try:
        rsp = client.PredictImages(req)
        # res_path = "/opt/tiger/workspace/datasets/self_test_datasets/ocr_res/" + cur_img_name + ".txt"
        # fp_w = open(res_path, "w")
        for i in range(len(rsp.results[0].words)):
            if np.random.random() <= random_value:  # 是否对当前文本行做ps
                # x表示width，y表示height
                x0 = str(rsp.results[0].words[i].det_points_abs[0].x)
                y0 = str(rsp.results[0].words[i].det_points_abs[0].y)
                x1 = str(rsp.results[0].words[i].det_points_abs[1].x)
                y1 = str(rsp.results[0].words[i].det_points_abs[1].y)
                x2 = str(rsp.results[0].words[i].det_points_abs[2].x)
                y2 = str(rsp.results[0].words[i].det_points_abs[2].y)
                x3 = str(rsp.results[0].words[i].det_points_abs[3].x)
                y3 = str(rsp.results[0].words[i].det_points_abs[3].y)
                '''
                至此，得到了当前图片的文字bbox，下面针对文字的bbox进行篡改：resize以及shift
                '''
                x0, y0, x1, y1, x2, y2, x3, y3 = int(eval(x0)), int(
                    eval(y0)), int(eval(x1)), int(eval(y1)), int(
                        eval(x2)), int(eval(y2)), int(eval(x3)), int(eval(y3))
                bbox_x1 = min(x0, x1, x2, x3)
                bbox_y1 = min(y0, y1, y2, y3)
                bbox_x2 = max(x0, x1, x2, x3)
                bbox_y2 = max(y0, y1, y2, y3)
                bbox_w = bbox_x2 - bbox_x1 + 1
                bbox_h = bbox_y2 - bbox_y1 + 1
                text = rsp.results[0].words[i].text
                if text == '':
                    text = '!None!'
                save_str = text + "\t" + str(bbox_x1) + "," + str(bbox_y1) + "," + str(bbox_x2) + "," + str(bbox_y1) + "," + \
                                str(bbox_x2) + "," + str(bbox_y2) + "," + str(bbox_x1) + "," + str(bbox_y2)
                # save_str = text + "\t" + x0 + "," + y0 + "," + x1 + "," + y1 + "," + x2 + "," + y2 + "," + x3 + "," + y3
                # fp_w.write(save_str)
                # fp_w.write("\n")
                ocr_det_res.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])
        # fp_w.close()
        return ocr_det_res
    except Exception as e:
        print("Wrong", e)


def resize_shift(cur_img,
                mask,
                cur_ocr_result_list,
                shift_ratio,
                resize_ratio,
                return_ratio=False):
    '''
        根据OCR检测区域对图片进行小范围的随机缩放与抖动
    '''
    cur_h, cur_w = cur_img.shape[:2]
    # mask = np.zeros([cur_h, cur_w], np.uint8)

    for line_idx in range(0, len(cur_ocr_result_list)):
        cur_line_bbox = cur_ocr_result_list[line_idx]
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = cur_line_bbox
        bbox_w = bbox_x2 - bbox_x1 + 1
        bbox_h = bbox_y2 - bbox_y1 + 1

        if np.random.random() < 0.5:
            # 文本框形状多变，设置像素点偏移或许可行，这里按照检测框的尺寸比例做shift
            # 很多时候可能文本是长条状的
            shift_x_ratio = random.uniform(shift_ratio[0], shift_ratio[1])
            shift_y_ratio = random.uniform(shift_ratio[0], shift_ratio[1])
            shift_x = int(shift_x_ratio * bbox_w)
            shift_y = int(shift_y_ratio * bbox_h)
            # 允许向左上方进行shift
            if np.random.random() < 0.5:
                shift_x = -shift_x
            if np.random.random() < 0.5:
                shift_y = -shift_y

            # 执行文本区域resize操作
            text_region = cur_img[bbox_y1:bbox_y2 + 1, bbox_x1:bbox_x2 + 1, :]
            r_ratio_x = random.uniform(resize_ratio[0], resize_ratio[1])
            r_ratio_y = random.uniform(resize_ratio[0], resize_ratio[1])
            text_region = cv2.resize(
                text_region,
                (int(bbox_w * r_ratio_x), int(bbox_h * r_ratio_y)))
            region_h, region_w = text_region.shape[:2]

            # debug | bound overflow
            # print("cur_img.shape = ", cur_w, cur_h)
            # print("左上角 = ", bbox_x1, bbox_y1)
            # print("右下角 = ", bbox_x2, bbox_y2)

            # 可能存在出界情况
            left_up_x = min(max(0, bbox_x1 + shift_x), cur_w - 1)
            left_up_y = min(max(0, bbox_y1 + shift_y), cur_h - 1)
            right_down_x = min(max(0, bbox_x1 + region_w - 1 + shift_x),
                               cur_w - 1)
            right_down_y = min(max(0, bbox_y1 + region_h - 1 + shift_y),
                               cur_h - 1)

            # cur_img = transforms(image=cur_img)['image']
            cur_img[left_up_y: right_down_y + 1, left_up_x: right_down_x + 1, :] = \
                    text_region[0: right_down_y - left_up_y + 1, 0: right_down_x - left_up_x + 1, :]
            # 0, 1 两个类别，1表示被修改过
            mask[left_up_y:right_down_y + 1, left_up_x:right_down_x + 1] = 1
        else:
            # remove
            remove_mask = np.zeros([cur_h, cur_w], np.uint8)
            remove_mask[bbox_y1:bbox_y2 + 1, bbox_x1:bbox_x2 + 1] = 255
            mask[bbox_y1:bbox_y2 + 1, bbox_x1:bbox_x2 + 1] = 1
            cur_img = cv2.inpaint(cur_img, remove_mask, 7, cv2.INPAINT_NS)

    if return_ratio:
        bg_count = len(mask[mask == 0])
        fg_count = len(mask[mask != 0])
        all_count = mask.shape[0] * mask.shape[1]
        return cur_img, mask, fg_count / all_count
    return cur_img, mask, -1


def rand_bbox(size):

    W = size[1]
    H = size[0]

    cut_rat_w = random.random() * 0.1 + 0.05
    cut_rat_h = random.random() * 0.1 + 0.05

    cut_w = int(W * cut_rat_w)
    cut_h = int(H * cut_rat_h)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def copy_move(img: np.array, img2: np.array, msk: np.array):
    size = img.shape
    W = size[1]
    H = size[0]

    if img2 is None:
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape[:])

        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))

        img[bby1 + y_move:bby2 + y_move,
            bbx1 + x_move:bbx2 + x_move, :] = img[bby1:bby2, bbx1:bbx2]
        msk[bby1 + y_move:bby2 + y_move, bbx1 + x_move:bbx2 + x_move] = 1
        # img = cv2.rectangle(img,
        #                     pt1=[bby1 + y_move, bbx1 + x_move],
        #                     pt2=[bby2 + y_move, bbx2 + x_move],
        #                     color=(255, 0, 0),
        #                     thickness=5)
    else:
        img2 = cv2.resize(img2, (W, H))
        assert img.shape == img2.shape
        bbx1, bby1, bbx2, bby2 = rand_bbox(img2.shape[:])

        x_move = random.randrange(-bbx1, (W - bbx2))
        y_move = random.randrange(-bby1, (H - bby2))

        img[bby1 + y_move:bby2 + y_move,
            bbx1 + x_move:bbx2 + x_move, :] = img2[bby1:bby2, bbx1:bbx2]
        msk[bby1 + y_move:bby2 + y_move, bbx1 + x_move:bbx2 + x_move] = 1
        # img = cv2.rectangle(img,
        #                     pt1=[bby1 + y_move, bbx1 + x_move],
        #                     pt2=[bby2 + y_move, bbx2 + x_move],
        #                     color=(255, 0, 0),
        #                     thickness=3)

    return img, msk

def visualization(img, mask, save_path = "/opt/tiger/workspace/mmsegmentation/data/vis_res", count= 1):

    ori_img = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

    img = img[:, :, ::-1]
    img[..., 1] = np.where(mask == 1, 255, img[..., 1])
    vis_img = np.hstack((ori_img, img))
    
    cv2.imwrite(os.path.join(save_path, str(count) + '.jpg'), vis_img)



def RandomShiftRemove(image_dir, img_path, img, mask, vis = False):
    
    # 所有的随机参数都放在这里
    resize_ratio = [1.0, 1.2]
    shift_ratio = [0, 0.05]
    random_sample_ocr = 0.5
    copy_move_ratio = 0.5
    copy_move_number = 3


    # cur_img = img

    img_list = os.listdir(image_dir)




    cur_ocr_result_list = get_random_ocr_result(img_path, random_value=random_sample_ocr)
    # print(cur_ocr_result_list)
    # 随机读取一张图片
    copy_img_idx = random.randint(0, len(img_list) - 1)

    copy_img_path = os.path.join(image_dir, img_list[copy_img_idx])
    copy_img = cv2.imread(copy_img_path)
    copy_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2RGB)
    # copy_ocr_result_list = get_random_ocr_result(copy_img_path, random_value=random_sample_ocr)
    # print(i, ' vs. ', copy_img_idx)
    # 针对随机获取的ocr区域做resize 与 shift 篡改，并返回ps区域所占图片的比例
    cur_img, cur_mask, mask_ratio = resize_shift(img,
                                                mask.copy(),
                                                cur_ocr_result_list,
                                                shift_ratio,
                                                resize_ratio,
                                                return_ratio=False)

    copy_move_cnt = random.randint(1, copy_move_number)
    while copy_move_cnt:
        if np.random.random() < copy_move_ratio:
            if np.random.random() < 0.5:
                cur_img, cur_mask = copy_move(cur_img, copy_img, cur_mask)
            else:
                cur_img, cur_mask = copy_move(cur_img, None, cur_mask)
        copy_move_cnt -= 1
    # cur_img, cur_mask, ratio = random_remove(cur_img, random_remove, cur_ocr_result_list)
    if vis:
        visualization(cur_img, cur_mask, count = 2)
        visualization(img, mask, count = 1)

    return cur_img, cur_mask

    bg_count = len(cur_mask[cur_mask == 0])
    fg_count = len(cur_mask[cur_mask != 0])
    all_count = cur_mask.shape[0] * cur_mask.shape[1]
    print("mask ratio = ", fg_count / all_count)
    cv2.imwrite(save_path + '/img/private_' + cur_img_name, cur_img)
    cv2.imwrite(save_path + '/mask/private_' + cur_img_name[:-4] + '.png',
                cur_mask)


if __name__ == '__main__':
    image_dir = '/opt/tiger/workspace/mmsegmentation/data/train/img'
    img_path = "/opt/tiger/workspace/mmsegmentation/data/train/img/1.jpg"
    mask_path = '/opt/tiger/workspace/mmsegmentation/data/train/mask/1.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, 0)
    mask = np.clip(mask, 0, 1)
    print(np.sum(mask))
    img, mask = RandomShiftRemove(image_dir, img_path, img, mask, vis = True)
    # visualization(img, mask)