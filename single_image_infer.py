from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import cv2
from tqdm import tqdm
import os

config_file = "/opt/tiger/workspace/mmsegmentation/configs/segformer/segformer_mit-b5_1024x1024_40k_tianchi_aug_class_weight1_3_mosaic.py"
checkpoint_file = '/opt/tiger/workspace/mmsegmentation/work_dirs/segformer_mit-b5_1024x1024_40k_tianchi_aug_class_weight1_3_mosaic/iter_24000.pth'
model = init_segmentor(config_file, checkpoint_file, device='cuda:4')

# mode = 'bytedance'
mode = 'submit'
if mode == 'bytedance':
    test_img_dir = 'PS_testset/img/'
else:
    test_img_dir = 'data/tianchi_aug/test/img/'

test_img_list = os.listdir(test_img_dir)
for img_name in tqdm(test_img_list):
    img = test_img_dir + img_name
    result = inference_segmentor(model, img)

    if mode == 'bytedance':
        img = show_result_pyplot(model, img, result, [
            [0, 0, 0], [255, 255, 255]], opacity=0.5)
        cv2.imwrite("PS_testset/segformer/"+img_name, img)
    else:
        result = result[0]
        result[:, :] = result[:, :] * 255
        cv2.imwrite("data/test_res_dir/images/"+img_name[:-4] + ".png", result)
