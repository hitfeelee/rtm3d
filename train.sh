#!/usr/bin/env bash



python train_multi_gpu.py --dist-url 'tcp://127.0.0.1:43458'  \
--dist-backend 'nccl' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
--model-config ./models/configs/rtm3d_resnet_kitti.yaml \
--ema \
--apex \
--test \
--opt-level 'O1' \
--loss-scale 'dynamic' # "large scale result in 'nan' "


