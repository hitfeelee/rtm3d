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
    for imgs, targets, paths, _ in pbar:
        src = imgs.clone().permute(1, 2, 0).contiguous().cpu().numpy()
        src = (src * dataset._norm_params['std_rgb'] + dataset._norm_params['mean_rgb']) * 255
        src = src.astype(np.uint8)
        imgs = imgs.unsqueeze(dim=0).to(cfg.DEVICE)
        with torch.no_grad():
            t1 = time.time()
            preds = model(imgs)[0]
            t2 = time.time()
        mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('%10s' + '%10.4g' * 2) % (mem, targets.get_field('mask').shape[0], t2-t1)
        pbar.set_description(s)
        clses, m_scores, m_projs, v_projs_regress, bboxes_2d = preds
        Ks = targets.get_field('K')

        src_vertex = np.copy(src)
        src_vertex_regress = np.copy(src)
        if clses[0] is not None:
            pred_out = ParamList.ParamList((0, 0))
            pred_out.add_field('class', clses[0])
            pred_out.add_field('score', m_scores[0])
            pred_out.add_field('bbox', bboxes_2d[0])
            out = model_utils.optim_decode_bbox3d(clses[0].cpu().numpy(),
                                                  v_projs_regress[0].cpu().numpy(),
                                                  Ks[0].cpu().numpy(),
                                                  cfg.DETECTOR.dim_ref, [0, -0.5, 20])
            visual_utils.cv_draw_main_kf(src,
                                         m_projs[0].cpu().numpy(),
                                         m_scores[0].cpu().numpy(),
                                         clses[0].cpu().numpy())
            visual_utils.cv_draw_bboxes_2d(src, pred_out,
                                           label_map=cfg.DATASET.OBJs)
            visual_utils.cv_draw_bbox3d_rtm3d(src_vertex_regress,
                                              clses[0].cpu().numpy(),
                                              m_scores[0].cpu().numpy(),
                                              v_projs_regress[0].cpu().numpy(),
                                              label_map=cfg.DATASET.OBJs
                                              )
            visual_utils.cv_draw_bboxes_3d_kitti(src_vertex, out,
                                                 label_map=cfg.DATASET.OBJs)
        # main_kf_logits, vertex_kf_logits = preds[:2]
        # main_kf_logits = main_kf_logits.sigmoid()
        # vertex_kf_logits = vertex_kf_logits.sigmoid()
        # main_kf = main_kf_logits[0].permute(1, 2, 0).contiguous().cpu().numpy() * 255
        # vertex_kf = vertex_kf_logits[0].permute(1, 2, 0).contiguous().cpu().numpy() * 255
        # h, w, c = vertex_kf.shape
        # vertex_kf = np.amax(vertex_kf.reshape(h, w, 3, -1), axis=-1)
        # h_src, w_src, _ = src.shape
        # main_kf = cv2.resize(main_kf.astype(np.uint8), (w_src, h_src))
        # vertex_kf = cv2.resize(vertex_kf.astype(np.uint8), (w_src, h_src))
        # main_heatmap = cv2.addWeighted(src, 1., main_kf, 1., 0)
        # vertex_heatmap = cv2.addWeighted(src, 1., vertex_kf, 1., 0)
        # heatmap = np.concatenate([main_heatmap, vertex_heatmap], axis=0)
        kf = np.concatenate([src,src_vertex, src_vertex_regress], axis=0)
        kf = cv2.resize(kf, (kf.shape[1] // 2, kf.shape[0] // 2))
        cv2.imshow('heatmap', kf)
        key = cv2.waitKey(1000)
        if key & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RTM3D Detecting")
    parser.add_argument("--model-config", default="", help="specific model config path")
    args = parser.parse_args()
    model, dataset, cfg = setup(args)
    detect(model, dataset, cfg)