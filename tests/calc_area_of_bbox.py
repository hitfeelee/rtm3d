import argparse
from utils import utils
import yaml
from datasets.dataset_reader import DatasetReader
import os
from preprocess.data_preprocess import TestTransform
import random
import cv2
import numpy as np
import tqdm
from models.configs.detault import CONFIGS as config
from datasets.data.kitti.devkit_object import utils as kitti_utils
from fvcore.common.config import CfgNode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=str, default='./models/configs/rtm3d_dla34_kitti.yaml')
    args = parser.parse_args()
    # opt.config = utils.check_file(opt.config)  # check file
    cfg = config.clone()
    if len(args.model_config) > 0:
        cfg.merge_from_file(args.model_config)
    opt = CfgNode(args.__dict__)
    cfg.merge_from_other_cfg(opt)

    brg_mean = config.DATASET.MEAN
    dr = DatasetReader(config.DATASET.PATH,  cfg, TestTransform(cfg.INPUT_SIZE[0], mean=brg_mean))

    batch_size = min(1, len(dr))
    names = cfg.DATASET.OBJs
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    bboxes_merge = []
    for img, target, path, _ in tqdm.tqdm(dr):
        bboxes_3d_array = target.numpy()
        bboxes = bboxes_3d_array.get_field('bbox')
        bboxes_merge.append(bboxes)

    bboxes = np.concatenate(bboxes_merge, axis=0)
    w = (bboxes[:, 2] - bboxes[:, 0])
    h = (bboxes[:, 3] - bboxes[:, 1])
    areas = w * h
    max_area = np.amax(areas)
    min_area = np.amin(areas)
    indx = np.argmax(areas)
    bbox = bboxes[indx]
    print('max area: %s, min area: %s' % (max_area, min_area))
