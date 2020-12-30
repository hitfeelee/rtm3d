#!/usr/bin/env bash
echo $1

if [[ $1 == '--multi-gpu' ]] ;then
  python train_multi_gpu.py --dist-url 'tcp://127.0.0.1:23456'  \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --model-config ./models/configs/rtm3d_resnet18_kitti.yaml
else
  python train.py --model-config ./models/configs/rtm3d_resnet18_kitti.yaml
fi

