import argparse
import os
from models import model_factory
from models.configs.detault import CONFIGS as config
from datasets.dataset_reader import DatasetReader, create_dataloader
from preprocess.data_preprocess import TestTransform
import torch
from utils import check_point
import tqdm
import cv2
import numpy as np
import time
from utils import visual_utils
from utils import model_utils
from utils import ParamList
from datasets.data.kitti.devkit_object import utils as kitti_utils
from fvcore.common.config import CfgNode

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup(args):
    cfg = config.clone()
    if len(args.model_config) > 0:
        cfg.merge_from_file(args.model_config)
    device = torch.device(cfg.DEVICE) if torch.cuda.is_available() else torch.device('cpu')
    cfg.update({'DEVICE': device})
    opt = CfgNode(args.__dict__)
    cfg.merge_from_other_cfg(opt)
    model = model_factory.create_model(cfg)
    dataset = DatasetReader(cfg.DATASET.PATH, cfg,
                            augment=TestTransform(cfg.INPUT_SIZE[0]), is_training=False, split='test')
    dataloader, _, dr = create_dataloader(config.DATASET.PATH, cfg,
                                          transform=TestTransform(cfg.INPUT_SIZE[0]),
                                          is_training=True, split='test')
    model.to(device)
    model.eval()
    return model, dataset, dataloader, cfg


def evaluate(model, dataset, dataloader, cfg):
    # save_dir = os.path.join(cfg.TRAINING.WEIGHTS, cfg.MODEL.BACKBONE)
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    checkpointer = check_point.CheckPointer(model,
                                            # save_dir=save_dir,
                                            mode='state-dict',
                                            device=cfg.DEVICE)

    ckpt = checkpointer.load(cfg.DETECTOR.CHECKPOINT, use_latest=False)
    del ckpt
    nb = len(dataloader)
    pbar = tqdm.tqdm(dataloader, total=nb)  # progress bar
    print(('\n' + '%10s' * 3) % ('mem', 'targets', 'time'))
    # half = cfg.DEVICE.type != 'cpu'
    half = False
    if half:
        model.half()

    for imgs, targets, paths, _, indexs in pbar:
        imgs = imgs.to(cfg.DEVICE)
        img_ids = targets.get_field('img_id')
        Ks = targets.get_field('K')
        Bs = imgs.shape[0]
        NKs = [None] * Bs
        for i in range(Bs):
            NKs[i] = Ks[img_ids == i][0:1, :]
        NKs = torch.cat(NKs, dim=0)
        NKs = NKs.to(cfg.DEVICE)
        invKs = NKs.view(-1, 3, 3).inverse()
        if half:
            imgs = imgs.half()
            invKs = invKs.half()
        with torch.no_grad():
            t1 = time.time()
            preds = model(imgs, invKs=invKs)[0]
            t2 = time.time()
        mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('%10s' + '%10.4g' * 2) % (mem, targets.get_field('mask').shape[0], t2-t1)
        pbar.set_description(s)
        for pred, index in zip(preds, indexs):
            results = []
            if pred is not None:
                pred = pred.cpu().numpy()
                index = index.cpu().item()
                src = dataset.load_image(index)
                clses = pred[:, 0].astype(np.int32)
                alphas = pred[:, 1]
                bboxes = pred[:, 2:6]
                dims = pred[:, 6:9]
                locs = pred[:, 9:12]
                Rys = pred[:, 12]
                scores = pred[:, 13]
                Ks = dataset.Ks[index].reshape(-1, 3, 3)
                for cls, alpha, bbox, dim, loc, ry, score in zip(clses, alphas, bboxes, dims, locs, Rys, scores):
                    l = ('%s ' * 15 + '%s\n') % (cfg.DATASET.OBJs[cls], 0, 0, alpha, *bbox, *dim, *loc, ry, score)
                    results.append(l)
                # visulal
                if cfg.visual:
                    pred_out = ParamList.ParamList((0, 0))
                    pred_out.add_field('class', clses)
                    pred_out.add_field('alpha', alphas)
                    pred_out.add_field('dimension', dims)
                    pred_out.add_field('location', locs)
                    pred_out.add_field('Ry', Rys)
                    pred_out.add_field('score', scores)
                    pred_out.add_field('K', Ks.repeat(len(clses), axis=0))
                    visual_utils.cv_draw_bboxes_3d_kitti(src, pred_out,
                                                         label_map=cfg.DATASET.OBJs)
                    cv2.imshow('test detection', src)
                    key = cv2.waitKey(1)
                    if key & 0xff == ord('q'):
                        break
            else:
                results.append(('%s ' * 15 + '%s\n') % ('DontCare', 0, 0, -10, 0, 0, 0, 0, 0, 0, 0, -1000, -1000, 0, 0, 0))
            with open(os.path.join(cfg.out_path, 'data', dataset.image_files[index] + '.txt'), 'w') as f:
                f.writelines(results)
    if cfg.visual:
        cv2.destroyAllWindows()
    print('start evaluate...')
    mAP = kitti_utils.evaluate('./datasets/data/kitti/training/label_2', cfg.out_path)
    return mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RTM3D Detecting")
    parser.add_argument("--model-config", default="", help="specific model config path")
    parser.add_argument("--out-path", default="./tests/tmp", help="specific output path")
    parser.add_argument('--num-workers', default=20, type=int,
                        help='the num of threads for data.')
    parser.add_argument('--visual', action='store_true',
                        help='visualize the detection result.')
    args = parser.parse_args()
    model, dataset, dataloader, cfg = setup(args)
    max_mAP = 25.5149
    best_model = None
    out_path = cfg.out_path
    for model_index in range(140, 141, 1):
        # model_path = './weights_multi_gpu/RESNET-18/model_{:07d}.pt'.format(model_index)
        model_path = 'models/pretrained/smoke/RESNET-18/model_best.pt'
        cfg.DETECTOR.CHECKPOINT = model_path
        # cfg.out_path = os.path.join(out_path, str(model_index))
        cfg.out_path = os.path.join(out_path, 'best')
        if not os.path.exists(cfg.out_path):
            os.mkdir(cfg.out_path)
            os.mkdir(os.path.join(cfg.out_path, 'data'))
        mAP = evaluate(model, dataset, dataloader, cfg)

        if max_mAP < mAP:
            max_mAP = mAP
            best_model = model_path
        print('{} mAP: {}'.format(model_path, mAP))
        print('best mAP: ', max_mAP)
    print('best model: ', best_model)
    print('best mAP: ', max_mAP)