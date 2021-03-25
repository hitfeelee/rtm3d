import argparse
import os
from models import model_factory
from models.configs.detault import CONFIGS as config
from datasets.dataset_seq_reader import SeqDatasetReader
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
from fvcore.common.config import CfgNode
from models.mot.AB3DMOT_libs.model import AB3DMOT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup(args):
    cfg = config.clone()
    if len(args.model_config) > 0:
        cfg.merge_from_file(args.model_config)
    opt = CfgNode(args.__dict__)
    cfg.merge_from_other_cfg(opt)
    device = torch.device(cfg.DEVICE) if torch.cuda.is_available() else torch.device('cpu')
    cfg.update({'DEVICE': device})
    model = model_factory.create_model(cfg, is_training=True)
    dataset = SeqDatasetReader(cfg.DATASET.PATH, cfg,
                            augment=TestTransform(cfg.INPUT_SIZE[0]), is_training=False, split='testing')
    model.to(device)
    model.eval()
    return model, dataset, cfg


def detect(model, dataset, cfg):
    checkpointer = check_point.CheckPointer(model,
                                            # save_dir=save_dir,
                                            mode='state-dict',
                                            device=cfg.DEVICE)

    ckpt = checkpointer.load(cfg.DETECTOR.CHECKPOINT, use_latest=False)
    del ckpt

    dataset.set_start_index(0)
    nb = len(dataset)
    pbar = tqdm.tqdm(dataset, total=nb)  # progress bar
    print(('\n' + '%10s' * 3) % ('mem', 'targets', 'time'))
    if cfg.record:
        encode = {'mp4': 'mp4v', 'avi': 'MJPG'}
        videowriter = cv2.VideoWriter('rtm3d_detect{}.{}'.format('_track' if cfg.tracking else '', cfg.video_format),
                                      cv2.VideoWriter.fourcc(*encode[cfg.video_format]), 20, (822, 208), True)
    half = cfg.DEVICE.type != 'cpu'
    # half = False
    # model.fuse()
    if half:
        model.half()
    mot_tracker = AB3DMOT()
    for imgs, targets, indexs in pbar:
        src = dataset.load_image(indexs)
        imgs = imgs.unsqueeze(dim=0).to(cfg.DEVICE)

        Ks = targets.get_field('K').to(cfg.DEVICE).view(-1, 3, 3)
        invKs = Ks.inverse()
        if half:
            imgs = imgs.half()
            invKs = invKs.half()

        with torch.no_grad():
            t1 = time.time()
            preds = model(imgs, invKs=invKs)[0]
            t2 = time.time()
        mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('%10s' + '%10.4g' * 2) % (mem, targets.get_field('K').shape[0], t2-t1)
        pbar.set_description(s)
        H, W, _ = src.shape
        ratio = 70./80.
        bird_view = np.zeros((H, int(H * ratio), 3), dtype=np.uint8)
        src_bv = np.copy(bird_view)
        if preds[0] is not None:
            K = dataset.load_calib_param(indexs)[0].reshape(-1, 3, 3)
            pred = preds[0].cpu().numpy()
            if not cfg.tracking:
                pred_out = ParamList.ParamList((0, 0))
                pred_out.add_field('class', pred[:, 0].astype(np.int32))
                pred_out.add_field('alpha', pred[:, 1])
                pred_out.add_field('dimension', pred[:, 2:5])
                pred_out.add_field('location', pred[:, 5:8])
                pred_out.add_field('Ry', pred[:, 8])
                pred_out.add_field('score', pred[:, 9])
                pred_out.add_field('K', K.reshape(1, 9).repeat((pred.shape[0]), axis=0))
                visual_utils.cv_draw_bboxes_3d_kitti(src, pred_out,
                                                     label_map=cfg.DATASET.OBJs, show_depth=True)
                # visual_utils.cv_draw_bbox3d_birdview(src_bv, targ, color=(0, 0, 255))
                visual_utils.cv_draw_bbox3d_birdview(src_bv, pred_out, color=(0, 255, 0), scaleX=0.2, scaleY=0.2)
            else:
                dets = pred[:, 2:9]
                info = np.concatenate([pred[:, :2], pred[:, -1:]], axis=-1)
                dets_all = {'dets': dets, 'info': info}
                trackers = mot_tracker.update(dets_all)
                if len(trackers) > 0:
                    pred_track = ParamList.ParamList((0, 0))
                    pred_track.add_field('dimension', trackers[:, 0:3])
                    pred_track.add_field('location', trackers[:, 3:6])
                    pred_track.add_field('Ry', trackers[:, 6])
                    pred_track.add_field('ID', trackers[:, 7])
                    pred_track.add_field('class', trackers[:, 8].astype(np.int32))
                    pred_track.add_field('alpha', trackers[:, 9])
                    pred_track.add_field('score', trackers[:, 10])
                    pred_track.add_field('K', K.reshape(1, 9).repeat((trackers.shape[0]), axis=0))

                    visual_utils.cv_draw_bboxes_3d_kitti(src, pred_track,
                                                         label_map=cfg.DATASET.OBJs, show_depth=True, show_id=True)
                    # visual_utils.cv_draw_bbox3d_birdview(src_bv, targ, color=(0, 0, 255))
                    visual_utils.cv_draw_bbox3d_birdview(src_bv, pred_track, color=(0, 255, 0), scaleX=0.2, scaleY=0.2)

        kf = np.concatenate([src, src_bv], axis=1)
        kf = cv2.resize(kf, (int(kf.shape[1] * 0.7), (int(kf.shape[0]*0.7))))
        cv2.imshow('rtm3d_detect', kf)
        if cfg.record:
            videowriter.write(kf)
        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()
    if cfg.record:
        videowriter.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RTM3D Detecting")
    parser.add_argument("--model-config", default="", help="specific model config path")
    parser.add_argument("--record", action='store_true', help="record detection result as video")
    parser.add_argument("--video_format", type=str, default='mp4', help="specific video formate, such as 'mp4', 'avi' ")
    parser.add_argument("--tracking", action='store_true', help="applying tracking")
    args = parser.parse_args()
    model, dataset, cfg = setup(args)
    detect(model, dataset, cfg)