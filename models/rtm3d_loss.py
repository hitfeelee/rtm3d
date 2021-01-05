import torch
import torch.nn as nn
from utils import model_utils
from utils import data_utils
from models.nets.module import FocalLoss, FocalLoss1
import torch.nn.functional as F
from preprocess.transforms import ToAbsoluteCoords
import numpy as np
from datasets.data.kitti.devkit_object import utils as kitti_utils
from models.smoke_decoder import SMOKECoder
import math
import time


class RTM3DLoss(nn.Module):
    def __init__(self, config):
        super(RTM3DLoss, self).__init__()
        self._config = config
        self._main_kf_loss = FocalLoss(config.MODEL.FOCAL_LOSS_ALPHA, config.MODEL.FOCAL_LOSS_BEDA)
        self._vertex_kf_loss = FocalLoss(config.MODEL.FOCAL_LOSS_ALPHA, config.MODEL.FOCAL_LOSS_BEDA)
        self._gaussian_scale = (config.DATASET.GAUSSIAN_SIGMA_MAX - config.DATASET.GAUSSIAN_SIGMA_MIN) / \
                               (config.DATASET.BBOX_AREA_MAX - config.DATASET.BBOX_AREA_MIN)
        self._ref_offset_fr_main = torch.tensor([self._config.DATASET.VERTEX_OFFSET_INFER])
        self._to_abs_coords = ToAbsoluteCoords()
        self._smode_ecoder = SMOKECoder(config)
        self._inited = False

    def test_build_main_targets(self, main_kf_logits, targets):
        self.__target_fit_pred_scale([main_kf_logits], targets)
        targets.apply(lambda x: x.type_as(main_kf_logits))
        return self.__build_main_targets(main_kf_logits, targets)

    def test_build_vertex_targets(self, vertex_kf_logits, targets):
        self.__target_fit_pred_scale([vertex_kf_logits], targets)
        targets.apply(lambda x: x.type_as(vertex_kf_logits))
        return self.__build_vertex_targets(vertex_kf_logits, targets)

    def test_build_targets(self, pred_logits, targets):

        return self._build_targets(pred_logits, targets)

    def __target_fit_pred_scale(self, pred_logits, targets):
        device = pred_logits[0].device
        H, W = pred_logits[0].shape[2:]
        img = np.zeros((H, W, 1), dtype=np.uint8)
        params = {}
        _, targets = self._to_abs_coords(img, targets, **params)
        if not self._inited:
            self._ref_offset_fr_main = self._ref_offset_fr_main.to(device)
            self._gaussian_scale /= (H * W)
            self._ref_offset_fr_main *= torch.tensor([[W, H]]).type_as(pred_logits[0]).to(device)
            self._inited = True
        return targets

    # @torch.no_grad()
    def __build_main_targets(self, main_kf_logits, targets):
        main_kf_mask = torch.zeros_like(main_kf_logits).detach()
        mask = targets.get_field('mask').clone().long()
        N = mask.shape[0]
        if mask.float().sum() <= 0:
            return main_kf_mask, torch.zeros((N, 2)).type_as(main_kf_mask).to(
                main_kf_mask.device), torch.zeros((N, 2)).type_as(main_kf_mask).to(main_kf_mask.device)
        bboxes = targets.get_field('bbox').clone()
        classes = targets.get_field('class').clone().long()
        img_id = targets.get_field('img_id').clone().long()
        noise_mask = targets.get_field('noise_mask').clone().bool()
        centers = data_utils.bbox_center(bboxes)

        if self._config.DATASET.GAUSSIAN_GEN_TYPE == 'dynamic_radius':
            gaussian_sigma, gaussian_radius, max_radius = self._dynamic_radius(bboxes)
        else:
            gaussian_sigma, gaussian_radius, max_radius = self._dynamic_sigma(bboxes)
        M = (2 * max_radius + 1) ** 2
        gaussian_kernel, offset_xy = model_utils.gaussian2D(gaussian_sigma, gaussian_radius, max_radius)
        gaussian_kernel[noise_mask, M // 2] = 0.9999
        N_indices = torch.arange(N).to(main_kf_mask.device).view(N, 1).repeat(1, M).flatten()  # (N x M, )
        target_xy = (centers[N_indices] + offset_xy.view(-1, 2)).long()  # (N x M , 2)
        img_id = img_id[N_indices]  # (N x M, )
        classes = classes[N_indices]  # (N x M, )
        H, W = main_kf_mask.shape[2:]
        main_kf_mask = torch.unsqueeze(main_kf_mask, dim=-1).repeat(1, 1, 1, 1, N)
        keep = (target_xy[:, 0] >= 0) & (target_xy[:, 0] < W) & (target_xy[:, 1] >= 0) & (
                target_xy[:, 1] < H)
        main_kf_mask[img_id[keep], classes[keep],
                     target_xy[keep][:, 1], target_xy[keep][:, 0], N_indices[keep]] = gaussian_kernel.flatten()[keep]
        main_kf_mask = torch.max(main_kf_mask, dim=-1)[0]
        proj_main = centers.long()
        offset_main = (centers - proj_main).type_as(main_kf_logits)
        return main_kf_mask, proj_main, offset_main

    # @torch.no_grad()
    def __build_vertex_targets(self, vertex_kf_logits, targets):
        vertex_kf_mask = torch.zeros_like(vertex_kf_logits)  #bs, C, H, W
        mask_3d = targets.get_field('mask_3d').clone()
        if mask_3d.sum() <= 0:
            return vertex_kf_mask, None, None, None
        C, H, W = vertex_kf_mask.shape[1:]
        device = vertex_kf_logits.device
        mask_m = targets.get_field('mask').clone().bool()
        img_id = targets.get_field('img_id').clone().long()
        bboxes = targets.get_field('bbox').clone()
        noise_mask = targets.get_field('noise_mask').clone().bool()
        vertexs = targets.get_field('vertex').clone()
        img_id_mask = img_id[mask_3d]
        bboxes_mask = bboxes[mask_3d]
        noise_mask_mask = noise_mask[mask_3d] | mask_m[mask_3d].bitwise_not()
        vertexs_mask = vertexs[mask_3d]
        # build mask
        N = vertexs_mask.shape[0]
        if self._config.DATASET.GAUSSIAN_GEN_TYPE == 'dynamic_radius':
            gaussian_sigma, gaussian_radius, max_radius = self._dynamic_radius(bboxes_mask)
        else:
            gaussian_sigma, gaussian_radius, max_radius = self._dynamic_sigma(bboxes_mask)
        M = (2 * max_radius + 1) ** 2
        gaussian_kernel, offset_xy = model_utils.gaussian2D(gaussian_sigma.view(-1, 1).repeat(1, C).flatten(),
                                                            gaussian_radius.view(-1, 1).repeat(1, C).flatten(),
                                                            max_radius)
        noise_mask_mask = noise_mask_mask.view(N, 1).repeat(1, C)
        gaussian_kernel[noise_mask_mask.flatten(), M // 2] = 0.9999
        N_indices = torch.arange(N).to(device).view(N, 1, 1).repeat(1, C, M).flatten()  # (N x C x M, )
        C_indices = torch.arange(C).to(device).view(1, C, 1).repeat(N, 1, M).flatten()  # (N x C x M, )

        offset_xy = offset_xy.view(-1, 2)
        target_xy = (vertexs_mask[N_indices, C_indices] + offset_xy).long()
        img_id_mask = img_id_mask.view(N, 1).repeat(1, C)[N_indices, C_indices]  # N x C x M

        vertex_kf_mask = torch.unsqueeze(vertex_kf_mask, dim=-1).repeat(1, 1, 1, 1, N)
        keep = (target_xy[:, 0] >= 0) & (target_xy[:, 0] < W) & (target_xy[:, 1] >= 0) & (
                target_xy[:, 1] < H)
        vertex_kf_mask[img_id_mask[keep], C_indices[keep],
                       target_xy[keep][:, 1], target_xy[keep][:, 0], N_indices[keep]] = gaussian_kernel.flatten()[keep]
        vertex_kf_mask = torch.max(vertex_kf_mask, dim=-1)[0]
        # build other offsets
        proj_vertex = vertexs.long()
        offset_vertex = (vertexs - proj_vertex).type_as(vertex_kf_logits)
        offset_fr_main = (vertexs - data_utils.bbox_center(bboxes).view(-1, 1, 2)).type_as(vertex_kf_logits)
        return vertex_kf_mask, offset_fr_main, proj_vertex, offset_vertex

    def _build_targets(self, pred_logits, targets):
        targets.apply(lambda x: x.type_as(pred_logits[0]))
        H, W = pred_logits[0].shape[2:]
        targets = targets.to(pred_logits[0].device)
        targets = self.__target_fit_pred_scale(pred_logits, targets)
        locations = targets.get_field('location').clone()
        Rys = targets.get_field('Ry').clone()
        dimensions = targets.get_field('dimension').clone()
        Ks = targets.get_field('K').clone()
        vertexs, _, mask_3d = kitti_utils.calc_proj2d_bbox3d(dimensions.cpu().numpy(),
                                                             locations.cpu().numpy(),
                                                             Rys.cpu().numpy(),
                                                             Ks.reshape(-1, 3, 3).cpu().numpy())
        vertexs = torch.from_numpy(vertexs[..., :-1]).to(Ks.dtype).to(Ks.device)  # get 8 vertex
        vertexs = vertexs.permute(0, 2, 1).contiguous()
        mask_3d = torch.from_numpy(mask_3d).to(Ks.device)
        mask_ver = (vertexs[..., 0] >= 0) & (vertexs[..., 0] < W) & (vertexs[..., 1] >= 0) & (
                vertexs[..., 1] < H)
        targets.add_field('vertex', vertexs)
        targets.add_field('mask_3d', mask_3d)
        targets.add_field('mask_ver', mask_ver)

        main_kf_logits, vertex_kf_logits, _, _, _ = pred_logits
        main_targets = self.__build_main_targets(main_kf_logits, targets)
        vertex_targets = self.__build_vertex_targets(vertex_kf_logits, targets)

        return main_targets, vertex_targets, targets

    def __call__(self, pred_logits, targets):
        # prediction logits
        m_hm_pred = pred_logits[0]
        regress_logits = pred_logits[1]
        m_off_pred = regress_logits[:, :2, :, :]
        pred3d_logits = regress_logits[:, 2:, :, :]

        # ground truth targets
        m_hm = targets.get_field('m_hm')
        m_projs = targets.get_field('m_proj').long()
        m_offs = targets.get_field('m_off')
        ver_coor = targets.get_field('v_coor_off')
        # vertexs = targets.get_field('vertex')
        img_id = targets.get_field('img_id').long()
        m_mask = targets.get_field('mask').bool()
        not_noise_mask = targets.get_field('noise_mask').bool().bitwise_not()
        v_mask = targets.get_field('v_mask').bool()

        # main kf loss
        loss_main_kf = self._main_kf_loss(model_utils.sigmoid_hm(m_hm_pred), m_hm)

        # main proj offset loss
        m_valid = m_mask & not_noise_mask
        m_off_pred = m_off_pred.permute(0, 2, 3, 1).contiguous()
        pos_main_offset = m_off_pred[img_id[m_valid],
                                     m_projs[m_valid][:, 1],
                                     m_projs[m_valid][:, 0]].sigmoid()
        loss_main_offset = F.l1_loss(pos_main_offset, m_offs[m_valid], reduction='mean')

        # 3d properties loss
        codes = self._smode_ecoder.encode_smoke_pred3d_and_targets(pred3d_logits, targets)
        pred_dims, pred_depths, pred_orients = codes[:3]
        gt_dims, gt_depths, gt_alphas = codes[-3:]
        loss_dim = F.l1_loss(pred_dims, gt_dims, reduction='mean')
        loss_depth = F.l1_loss(pred_depths, gt_depths, reduction='mean')
        loss_orient = -1. * torch.cos(torch.atan2(pred_orients[:, 0],
                                                 pred_orients[:, 1]) -
                                     gt_alphas).mean() + 1.

        loss_main_kf *= self._config.TRAINING.W_MKF
        loss_main_offset *= self._config.TRAINING.W_M_OFF
        loss_dim *= self._config.TRAINING.W_V_OFF
        loss_depth *= self._config.TRAINING.W_V_OFF
        loss_orient *= self._config.TRAINING.W_V_OFF

        loss = loss_main_kf + loss_main_offset + loss_dim + loss_depth + loss_orient
        return loss, torch.tensor([loss_main_kf, loss_main_offset, loss_dim, loss_depth, loss_orient, loss],
                                  device=loss.device).detach()


