import argparse
from utils import utils
import yaml
from datasets.dataset_reader import DatasetReader, create_dataloader
import os
from preprocess.data_preprocess import TrainAugmentation
import random
import cv2
import numpy as np
from utils import visual_utils
from models.configs.detault import CONFIGS as config
from models.rtm3d_loss import RTM3DLoss
import torch
from fvcore.common.config import CfgNode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=str, default='./models/configs/rtm3d_dla34_kitti.yaml')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='the num of threads for data.')
    args = parser.parse_args()
    # opt.config = utils.check_file(opt.config)  # check file
    cfg = config.clone()
    if len(args.model_config) > 0:
        cfg.merge_from_file(args.model_config)
    opt = CfgNode(args.__dict__)
    cfg.merge_from_other_cfg(opt)

    brg_mean = config.DATASET.MEAN
    dr = DatasetReader(config.DATASET.PATH,  cfg, TrainAugmentation(cfg.INPUT_SIZE[0], mean=brg_mean))
    dataloader, _, dr = create_dataloader(config.DATASET.PATH, cfg,
                                          transform=TrainAugmentation(cfg.INPUT_SIZE[0], mean=brg_mean), is_training=True)
    batch_size = min(1, len(dr))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    names = cfg.DATASET.OBJs
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    rtm3d_loss = RTM3DLoss(config)
    for img, target, path, _ in dataloader:
        img_name = path.split('/')[-1][:-4]
        target.to_tensor()
        h, w = img.shape[1:]
        src = img.permute(1, 2, 0).cpu().numpy()
        src = (src * dr._norm_params['std_rgb'] + dr._norm_params['mean_rgb']) * 255
        src = src.astype(np.uint8)
        src_ver = np.copy(src)
        pred = torch.zeros((1, 3, h, w)).type_as(img).to(img.device)
        pred_ver = torch.zeros((1, 8, h, w)).type_as(img).to(img.device)
        pred_logits = pred, pred_ver, None, None, None
        output = rtm3d_loss.test_build_targets(pred_logits, target)
        hm_m = output[0][0]
        hm_m = hm_m[0].permute(1, 2, 0).contiguous().cpu().numpy() * 255
        hm_m = hm_m.astype(np.uint8)
        src = cv2.addWeighted(src, 1., hm_m, 1., 0)
        # mask = hm_m > 0
        # src[mask] = 255
        ys, xs = np.nonzero(hm_m.max(axis=-1) == 255)
        for x, y in zip(xs, ys):
            cv2.circle(src, (x, y), 3, (0, 0, 255), thickness=-1)

        hm_ver = output[1][0]
        hm_ver = hm_ver[0].permute(1, 2, 0).contiguous().view(h, w, 1, -1)
        hm_ver = hm_ver.max(dim=-1)[0].cpu().numpy() * 255
        hm_ver = hm_ver.astype(np.uint8)
        hm_ver = np.dstack([hm_ver, np.zeros_like(hm_ver), np.zeros_like(hm_ver)])
        src_ver = cv2.addWeighted(src_ver, 1., hm_ver, 1., 0)

        ys, xs = np.nonzero(hm_ver.max(axis=-1) == 255)
        for x, y in zip(xs, ys):
            cv2.circle(src_ver, (x, y), 3, (0, 0, 255), thickness=-1)
        # cv2.putText(src, img_name, (50, 50), 0, 2, (0, 0, 255), 2)
        res = np.concatenate([src, src_ver], axis=0)
        cv2.imshow('test result', res)

        key = cv2.waitKey(3000)
        if key == 'q':
            break

