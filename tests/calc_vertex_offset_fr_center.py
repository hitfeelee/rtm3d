import argparse
from utils import utils
import yaml
from datasets.dataset_reader import DatasetReader, create_dataloader
import os
from preprocess.data_preprocess import TestTransform
import random
import cv2
import numpy as np
import tqdm
from models.configs.detault import CONFIGS as config
from datasets.data.kitti.devkit_object import utils as kitti_utils
from fvcore.common.config import CfgNode
from utils import visual_utils
from utils import ParamList

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=str, default='./models/configs/rtm3d_resnet_nuscene.yaml')
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
    # dr = DatasetReader(config.DATASET.PATH,  cfg, TestTransform(cfg.INPUT_SIZE[0], mean=brg_mean))
    dataloader, _, dr = create_dataloader(config.DATASET.PATH, cfg,
                        transform=TestTransform(cfg.INPUT_SIZE[0], mean=brg_mean), is_training=True,split='train')
    batch_size = min(1, len(dr))
    names = cfg.DATASET.OBJs
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    vertexs = []
    centers = []
    paths = []
    for img, target, path, _, index in tqdm.tqdm(dr):
        src = img.permute(1, 2, 0).cpu().numpy()
        src = (src * dr._norm_params['std_rgb'] + dr._norm_params['mean_rgb']) * 255
        src = src.astype(np.uint8)
        back_up = np.copy(src)

        bboxes_3d_array = target.numpy()
        classes = bboxes_3d_array.get_field('class').astype(np.int)
        N = len(classes)
        scores = bboxes_3d_array.get_field('score') if bboxes_3d_array.has_field('score') else np.ones((N,),
                                                                                                       dtype=np.int)
        locations = bboxes_3d_array.get_field('location')
        Rys = bboxes_3d_array.get_field('Ry')
        dimensions = bboxes_3d_array.get_field('dimension')
        Ks = bboxes_3d_array.get_field('K')
        Ks[:, :6] *= 4
        mask = bboxes_3d_array.get_field('mask').astype(np.bool)
        noise_mask = bboxes_3d_array.get_field('noise_mask').astype(np.bool)

        proj2des, bboxes_2d, mask_3d = kitti_utils.calc_proj2d_bbox3d(dimensions, locations, Rys, Ks.reshape(N, 3, 3),xrange=(-200, 1500), yrange=(-200, 500))
        bboxes = bboxes_3d_array.get_field('bbox') if bboxes_3d_array.has_field('bbox') else np.copy(bboxes_2d)
        # H, W = src.shape[:2]
        # bboxes_2d[:, 0::2] = np.clip(bboxes_2d[:, 0::2], a_min=0, a_max=W)
        # bboxes_2d[:, 1::2] = np.clip(bboxes_2d[:, 1::2], a_min=0, a_max=H)
        # iou = kitti_utils.box_iou(bboxes, bboxes_2d)
        # keep = iou < 0.4
        # keep = np.bitwise_not(mask)
        occ = kitti_utils.clean_full_occlusion_objects(np.concatenate([classes.astype(np.float).reshape(-1, 1),
                                                                       bboxes_2d,
                                                                       dimensions,
                                                                       bboxes_3d_array.get_field('alpha').reshape(-1, 1),
                                                                       Rys.reshape(-1, 1),
                                                                       locations
                                                                       ], axis=-1),th=0.9)
        keep = mask & mask_3d & np.bitwise_not(noise_mask)
        if keep.astype(np.long).sum() > 0:
            vertexs.append(proj2des[keep])
            center = np.hstack([(bboxes[:, None, 0] + bboxes[:, None, 2]) / 2,
                                (bboxes[:, None, 1] + bboxes[:, None, 3]) / 2])
            centers.append(center[keep])
            paths.append(np.array([path] * len(center[keep])))
            gt = ParamList.ParamList(bboxes_3d_array.size)
            gt.copy_field(bboxes_3d_array,['class', 'mask', 'dimension', 'location', 'Ry', 'K'])
            keep = mask & np.bitwise_not(noise_mask) #& occ
            gt.update_field('mask', keep)
            gt.delete_by_mask()
            visual_utils.cv_draw_bboxes_3d_kitti(back_up, gt, label_map=cfg.DATASET.OBJs)

        print(path)
        cv2.putText(back_up, dr.image_files[index], (100, 100),0, 1.,color=(255, 0, 0))
        cv2.imshow('check', back_up)
        key = cv2.waitKey(1000)
        if key & 0xff == ord('q'):
            break
    paths = np.concatenate(paths, axis=0)
    vertexs = np.concatenate(vertexs, axis=0)
    centers = np.concatenate(centers, axis=0)
    centers = centers.reshape((-1, 2, 1))
    offsets = vertexs - centers
    offsets = np.transpose(offsets, axes=[0, 2, 1]).reshape(-1, 2)
    max_off = np.amax(offsets, axis=0)
    min_off = np.amin(offsets, axis=0)
    max_indice = np.argmax(offsets, axis=0)
    print(paths[max_indice])
    print('max offset: %s, min offset: %s' % (max_off, min_off))
