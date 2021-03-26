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
from models.nets import repvgg
from models.model import Model
from models.configs.detault import CONFIGS as configs
try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    # raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    print("Not found apex module.")


def create_model(configs, is_training=True):
    """Create model based on architecture name"""

    if 'DLA-34' in configs.MODEL.BACKBONE:
        print('using DLA-34 architecture')
        backbone = dla.create_model(configs)
        model = Model(configs, backbone)
    elif 'RESNET' in configs.MODEL.BACKBONE:
        print('using RESNET-X architecture')
        backbone = resnet.get_pose_net(configs.MODEL.BACKBONE.split('-')[-1], configs)
        model = Model(configs, backbone)
    elif 'RepVGG' in configs.MODEL.BACKBONE:
        print('using RepVGG-X architecture')
        backbone = repvgg.get_RepVGG_func_by_name(configs.MODEL.BACKBONE)(configs, deploy=(not is_training))
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


def convert_repvgg_model_2_inference_time(configs, device, save_dir=None):
    model = create_model(configs)
    model.to(device)
    ckpts = torch.load(configs.TRAINING.CHECKPOINT_FILE)
    model.load_state_dict(ckpts["model"])
    repvgg_net = model.backbone
    repvgg_fun = repvgg.get_RepVGG_func_by_name(configs.MODEL.BACKBONE)
    repvgg_deploy = repvgg.repvgg_model_convert(repvgg_net, repvgg_fun, configs)
    model.backbone = repvgg_deploy
    params = {"model": model.state_dict()}
    if save_dir is not None:
        torch.save(params, save_dir)
    return model


def make_data_parallel(model, configs):
    if configs.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        # torch.cuda.set_device(configs.gpu_idx)
        # model.cuda(configs.gpu_idx).to(memory_format=configs.memory_format)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        if configs.apex:
            model = DDP(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[
                                                                  configs.gpu_idx] if configs.gpu_idx != -1 else None,
                                                              find_unused_parameters=True)
    else:
        model = model.cuda(configs.gpu_idx) if configs.gpu_idx != -1 else model.cuda()

    return model

import time

if __name__ == '__main__':
    import argparse
    from fvcore.common.config import CfgNode
    # from torchsummary import summary

    parser = argparse.ArgumentParser(description='RTM3D Implementation')
    parser.add_argument('--config', type=str, default='./models/configs/rtm3d_resnet18_kitti.yaml', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cmd', type=str, default='test',
                        help='specific the cmd to be run')
    opt = parser.parse_args()
    configs.merge_from_file(opt.config)
    args = CfgNode(opt.__dict__)
    configs.merge_from_other_cfg(args)
    device = torch.device('cuda')
    configs.update({'DEVICE': device})

    if opt.cmd == 'test':
        model = create_model(configs, is_training=False).to(device=device).float()
        sample_input = torch.randn((1, 3, 416, 1280)).to(device=device).to(torch.float32)
        # summary(model.cuda(1), (3, 224, 224))
        sums = 0
        for _ in range(20):
            t1 = time.time()
            output = model(sample_input)
            t2 = time.time()
            sums += (t2 - t1)
            print('inference time of model: ', t2 - t1)

        print('number of parameters: {}'.format(get_num_parameters(model)))
        print('inference time: ', sums/20)
    elif opt.cmd == 'repvgg_deploy':
        model_deploy = convert_repvgg_model_2_inference_time(configs, device,
                                                             './models/pretrained/simple-smoke/RepVGG-A1/model_best_deploy.pt')

        pass
