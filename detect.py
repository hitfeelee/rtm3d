import argparse
import os
from models import model_factory
from models.configs.detault import CONFIGS as config
from datasets.dataset_reader import DatasetReader
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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup(args):
    cfg = config.clone()
    if len(args.model_config) > 0:
        cfg.merge_from_file(args.model_config)
    device = torch.device(cfg.DEVICE) if torch.cuda.is_available() else torch.device('cpu')
    cfg.update({'DEVICE': device})
    model = model_factory.create_model(cfg)
    dataset = DatasetReader(cfg.DATASET.PATH, cfg,
                            augment=TestTransform(cfg.INPUT_SIZE[0]), is_training=False, split='test')
    model.to(device)
    model.eval()
    return model, dataset, cfg


def detect(model, dataset, cfg):
    # save_dir = os.path.join(cfg.TRAINING.WEIGHTS, cfg.MODEL.BACKBONE)
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    checkpointer = check_point.CheckPointer(model,
                                            # save_dir=save_dir,
                                            mode='state-dict',
                                            device=cfg.DEVICE)

    ckpt = checkpointer.load(cfg.DETECTOR.CHECKPOINT, use_latest=False)
    del ckpt
    nb = len(dataset)
    pbar = tqdm.tqdm(dataset, total=nb)  # progress bar
    print(('\n' + '%10s' * 3) % ('mem', 'targets', 'time'))
    # videowriter = cv2.VideoWriter('rtm3d_detect.mp4', cv2.VideoWriter.fourcc(*'mp4v'), 1, (848, 624),True)
    for imgs, targets, paths, _ in pbar:
        src = imgs.clone().permute(1, 2, 0).contiguous().cpu().numpy()
        src = (src * dataset._norm_params['std_rgb'] + dataset._norm_params['mean_rgb']) * 255
        src = src.astype(np.uint8)
        imgs = imgs.unsqueeze(dim=0).to(cfg.DEVICE)
        params = ParamList.ParamList(targets.size)
        img_ids = targets.get_field('img_id')
        Ks = targets.get_field('K')
        Bs = imgs.shape[0]
        NKs = [None] * Bs
        for i in range(Bs):
            NKs[i] = Ks[img_ids == i][0:1, :]
        params.add_field('K', torch.cat(NKs, dim=0))
        params = params.to(cfg.DEVICE)
        with torch.no_grad():
            t1 = time.time()
            preds = model(imgs, Ks=params.get_field('K'))[0]
            t2 = time.time()
        mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('%10s' + '%10.4g' * 2) % (mem, targets.get_field('mask').shape[0], t2-t1)
        pbar.set_description(s)
        Ks = params.get_field('K')
        H, W, _ = src.shape
        bird_view = np.zeros((H, H, 3), dtype=np.uint8)
        src_bv = np.copy(bird_view)
        src_optim = np.copy(src)
        src_optim_bv = np.copy(bird_view) + np.array([50, 50, 50], dtype=np.uint8)
        src_vertex_reg = np.copy(src)
        src_vertex_reg_bv = np.copy(bird_view) + np.array([100, 100, 100], dtype=np.uint8)
        if preds[0] is not None:
            K = Ks[0].cpu().numpy()
            K[:6] *= cfg.MODEL.DOWN_SAMPLE
            pred = preds[0].cpu().numpy()
            pred_out = ParamList.ParamList((0, 0))
            pred_out.add_field('class', pred[:, 0].astype(np.int32))
            pred_out.add_field('alpha', pred[:, 1])
            pred_out.add_field('bbox', pred[:, 2:6])
            pred_out.add_field('dimension', pred[:, 6:9])
            pred_out.add_field('location', pred[:, 9:12])
            pred_out.add_field('Ry', pred[:, 12])
            pred_out.add_field('score', pred[:, 13])
            pred_out.add_field('vertex', pred[:, 14:].reshape(-1, 8, 2))
            pred_out.add_field('K', K.reshape(1, 9).repeat((pred.shape[0]), axis=0))

            targ = ParamList.ParamList(targets.size, is_training=False)
            targ.copy_field(targets, ['mask', 'class', 'noise_mask',
                                         'dimension', 'location', 'Ry', 'alpha'])
            m_mask = targ.get_field('mask').bool()
            noise_mask = targ.get_field('noise_mask')
            m_mask &= noise_mask.bool().bitwise_not()
            targ.update_field('mask', m_mask)
            N = m_mask.float().sum()
            targ.delete_by_mask()
            targ = targ.numpy()
            targ.update_field('K', K.reshape(1, 9).repeat((N,), axis=0))
            optim_out = model_utils.optim_decode_bbox3d(pred_out, K)

            visual_utils.cv_draw_bboxes_3d_kitti(src, pred_out,
                                                 label_map=cfg.DATASET.OBJs)
            visual_utils.cv_draw_bbox3d_birdview(src_bv, pred_out, color=(255, 0, 0))
            visual_utils.cv_draw_bbox3d_birdview(src_bv, targ, color=(0, 0, 255))
            visual_utils.cv_draw_bboxes_3d_kitti(src_optim, optim_out,
                                                 label_map=cfg.DATASET.OBJs)
            visual_utils.cv_draw_bbox3d_birdview(src_optim_bv, optim_out, color=(255, 0, 0))
            visual_utils.cv_draw_bbox3d_birdview(src_optim_bv, targ, color=(0, 0, 255))
            visual_utils.cv_draw_bbox3d_rtm3d(src_vertex_reg,
                                              pred_out.get_field('class'),
                                              pred_out.get_field('score'),
                                              pred_out.get_field('vertex'),
                                              label_map=cfg.DATASET.OBJs)

        kf = np.concatenate([np.concatenate([src, src_bv], axis=1),
                             np.concatenate([src_optim, src_optim_bv], axis=1),
                             np.concatenate([src_vertex_reg, src_vertex_reg_bv], axis=1)], axis=0)
        kf = cv2.resize(kf, (kf.shape[1] // 2, kf.shape[0] // 2))
        cv2.imshow('rtm3d_detect', kf)
        # videowriter.write(kf)
        key = cv2.waitKey(100)
        if key & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()
    # videowriter.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RTM3D Detecting")
    parser.add_argument("--model-config", default="", help="specific model config path")
    args = parser.parse_args()
    model, dataset, cfg = setup(args)
    detect(model, dataset, cfg)