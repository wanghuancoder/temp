#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
export FLAGS_USE_STANDALONE_EXECUTOR=0
export FLAGS_eager_delete_tensor_gb=0.5


python3.7 ./ppcls/static/train.py -c ./ResNet50.yaml > ./new_executor_log/ResNet50_old_96.txt
python3.7 ./ppcls/static/train.py -c ./SE_ResNet50_vd.yaml > ./new_executor_log/SE_ResNet50_old_64.txt
python3.7 ./ppcls/static/train.py -c ./MobileNetV1.yaml > ./new_executor_log/MobileNetV1_old_64.txt
python3.7 ./ppcls/static/train.py -c ./AlexNet.yaml > ./new_executor_log/AlexNet_old_64.txt

python3.7 ./ppcls/static/train.py -c ./ResNet50_bs1.yaml > ./new_executor_log/ResNet50_old_1.txt
python3.7 ./ppcls/static/train.py -c ./SE_ResNet50_vd_bs1.yaml > ./new_executor_log/SE_ResNet50_old_1.txt
python3.7 ./ppcls/static/train.py -c ./MobileNetV1_bs1.yaml > ./new_executor_log/MobileNetV1_old_1.txt
python3.7 ./ppcls/static/train.py -c ./AlexNet_bs1.yaml > ./new_executor_log/AlexNet_old_1.txt


export FLAGS_USE_STANDALONE_EXECUTOR=1

python3.7 ./ppcls/static/train.py -c ./ResNet50.yaml > ./new_executor_log/ResNet50_new_96.txt
python3.7 ./ppcls/static/train.py -c ./SE_ResNet50_vd.yaml > ./new_executor_log/SE_ResNet50_new_64.txt
python3.7 ./ppcls/static/train.py -c ./MobileNetV1.yaml > ./new_executor_log/MobileNetV1_new_64.txt
python3.7 ./ppcls/static/train.py -c ./AlexNet.yaml > ./new_executor_log/AlexNet_new_64.txt

python3.7 ./ppcls/static/train.py -c ./ResNet50_bs1.yaml > ./new_executor_log/ResNet50_new_1.txt
python3.7 ./ppcls/static/train.py -c ./SE_ResNet50_vd_bs1.yaml > ./new_executor_log/SE_ResNet50_new_1.txt
python3.7 ./ppcls/static/train.py -c ./MobileNetV1_bs1.yaml > ./new_executor_log/MobileNetV1_new_1.txt
python3.7 ./ppcls/static/train.py -c ./AlexNet_bs1.yaml > ./new_executor_log/AlexNet_new_1.txt
