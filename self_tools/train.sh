./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]


# 单机多任务
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4


./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]

python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

python3 tools/test.py configs/segformer/segformer_mit-b5_1024x1024_40k_tianchi_aug.py work_dirs/segformer_mit-b5_1024x1024_40k_tianchi_aug/iter_8000.pth --out data/test_res_dir/test_res/2507.pkl --aug-test --format-only
./tools/dist_test.sh configs/segformer/segformer_mit-b5_1024x1024_40k_tianchi_aug.py work_dirs/segformer_mit-b5_1024x1024_40k_tianchi_aug/iter_8000.pth  8 --aug-test --format-only --eval-options "imgfile_prefix=./segformer_test_results"


./tools

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ./tools/dist_train.sh configs/segformer/segformer_mit-b5_1024x1024_40k_tianchi_aug_class_weight1_3.py 6 