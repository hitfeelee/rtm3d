#!/usr/bin/env bash


if [[ $1 == '--multi-gpu' ]] ;then
  echo 'multi gpu training...'
  python train_multi_gpu.py --dist-url 'tcp://127.0.0.1:23456'  \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --model-config ./models/configs/rtm3d_resnet18_kitti.yaml --test
else
  echo 'single gpu training...'
  python train.py --model-config ./models/configs/rtm3d_resnet18_kitti.yaml --test
fi
