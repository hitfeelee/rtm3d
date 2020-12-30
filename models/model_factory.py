"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: hitfee.li
# DoC: 2020.08.09
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: utils functions that use for model
"""

import sys

import torch

sys.path.append('/')

from models.nets import dla
from models.nets import resnet
from models.model import Model
from models.configs.detault import CONFIGS as configs


def create_model(configs):
    """Create model based on architecture name"""

    if 'DLA-34' in configs.MODEL.BACKBONE:
        print('using DLA-34 architecture')
        backbone = dla.create_model(configs)
        model = Model(configs, backbone)
    elif 'RESNET' in configs.MODEL.BACKBONE:
        print('using RESNET-X architecture')
        backbone = resnet.get_pose_net(configs.MODEL.BACKBONE.split('-')[-1], configs)
        model = Model(configs, backbone)
    else:
        assert False, 'Undefined model backbone'

    return model


def get_num_parameters(model):
    """Count number of trained parameters of the model"""
    if hasattr(model, 'module'):
        num_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_parameters


def make_data_parallel(model, configs):
    if configs.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if configs.gpu_idx != -1:
            torch.cuda.set_device(configs.gpu_idx)
            model.cuda(configs.gpu_idx)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            configs.BATCH_SIZE = int(configs.BATCH_SIZE / configs.ngpus_per_node)
            configs.num_workers = int((configs.num_workers + configs.ngpus_per_node - 1) / configs.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs.gpu_idx],
                                                              find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif configs.gpu_idx != -1:
        torch.cuda.set_device(configs.gpu_idx)
        model = model.cuda(configs.gpu_idx)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    return model

import time

if __name__ == '__main__':
    import argparse

    from torchsummary import summary


    parser = argparse.ArgumentParser(description='RTM3D Implementation')
    parser.add_argument('--config', type=str, default='./models/configs/rtm3d_resnet18_kitti.yaml', metavar='ARCH',
                        help='The name of the model architecture')

    opt = parser.parse_args()
    configs.merge_from_file(opt.config)
    device = torch.device('cpu')
    # configs.device = torch.device('cpu')

    model = create_model(configs).to(device=device).float()
    sample_input = torch.randn((1, 3, 416, 1280)).to(device=device).to(torch.float32)
    # summary(model.cuda(1), (3, 224, 224))
    for _ in range(10):
        t1 = time.time()
        output = model(sample_input)
        t2 = time.time()
        print('inference time of model: ', t2 - t1)

    print('number of parameters: {}'.format(get_num_parameters(model)))
