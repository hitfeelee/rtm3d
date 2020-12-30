import torch
import torch.nn as nn
from utils import model_utils
from utils import data_utils
from models.nets.module import FocalLoss, FocalLoss1
import torch.nn.functional as F
from preprocess.transforms import ToAbsoluteCoords
import numpy as np
from datasets.data.kitti.devkit_object import utils as kitti_utils
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
        self._inited = False

    def _compute_gaussian_radius(self, bboxes, min_overlap=0.7):
        height, width = torch.ceil(bboxes[:, 3] - bboxes[:, 1]), torch.ceil(bboxes[:, 2] - bboxes[:, 0])

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return torch.cat([r1.unsqueeze(dim=-1), r2.unsqueeze(dim=-1), r3.unsqueeze(dim=-1)], dim=-1).min(dim=-1)[0]

    def _dynamic_sigma(self, bboxes):
        areas = data_utils.bbox_area(bboxes)
        gaussian_sigma = torch.sqrt((areas - self._config.DATASET.BBOX_AREA_MIN) * self._gaussian_scale + \
                                    self._config.DATASET.GAUSSIAN_SIGMA_MIN)
        # gaussian_sigma = torch.sqrt(areas * self._gaussian_scale)
        gaussian_radius = gaussian_sigma * 3
        return gaussian_sigma, gaussian_radius, math.ceil(gaussian_radius.max().cpu().item())

    def _dynamic_radius(self, bboxes):
        gaussian_radius = self._compute_gaussian_radius(bboxes)
        gaussian_sigma = (2 * gaussian_radius + 1) / 6
        return gaussian_sigma, gaussian_radius, math.ceil(gaussian_radius.max().cpu().item())

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

    # def __call__(self, pred_logits, targets):
    #     main_targets, vertex_targets, targets = self._build_targets(pred_logits, targets) #, vertex_targets
    #     # prediction logits
    #     main_kf_logits, vertex_kf_logits = pred_logits[:2]
    #     offset_fr_main_logits, main_offset_kf_logits, vertex_offset_kf_logits = pred_logits[2:]
    #
    #     # ground truth targets
    #     main_kf_targets, proj_main, offset_main = main_targets
    #     vertex_kf_targets, offset_fr_main, proj_vertex, offset_vertex = vertex_targets
    #
    #     # main kf loss
    #     loss_main_kf = self._main_kf_loss(model_utils.sigmoid_hm(main_kf_logits), main_kf_targets)
    #
    #     # vertex kf loss
    #     loss_vertex_kf = self._vertex_kf_loss(model_utils.sigmoid_hm(vertex_kf_logits), vertex_kf_targets)
    #
    #     # offset from main loss and vertex proj offset loss
    #     device = main_kf_logits.device
    #     img_id = targets.get_field('img_id').clone().long()
    #     m_mask = targets.get_field('mask').clone().bool()
    #     not_noise_mask = targets.get_field('noise_mask').clone().bool().bitwise_not()
    #     mask_3d = targets.get_field('mask_3d').clone()
    #     mask_ver = targets.get_field('mask_ver').clone()
    #     B, C, H, W = offset_fr_main_logits.shape
    #     num_vc = C // 2
    #     assert num_vc == proj_vertex.shape[1]
    #     ofm_valid = (m_mask & not_noise_mask & mask_3d)
    #     ofm_valid_expand = mask_ver[ofm_valid].view(-1)
    #     # offset from main loss
    #     offset_fr_main_logits = offset_fr_main_logits.permute(0, 2, 3, 1).contiguous()
    #     pos_offset_fr_main = offset_fr_main_logits[img_id[ofm_valid],
    #                                                proj_main[ofm_valid][:, 1],
    #                                                proj_main[ofm_valid][:, 0]].view(-1, 2)
    #     pos_offset_fr_main = (2 * pos_offset_fr_main.sigmoid_() - 1.) * self._ref_offset_fr_main.type_as(
    #         pos_offset_fr_main).to(device)
    #     loss_offset_fr_main = F.l1_loss(pos_offset_fr_main[ofm_valid_expand],
    #                                            offset_fr_main[ofm_valid].view(-1, 2)[ofm_valid_expand], reduction='mean')
    #
    #     # vertex proj offset loss
    #     bs = img_id.view(-1, 1).repeat(1, num_vc).view(-1)
    #     proj_vertex = proj_vertex.view(-1, 2)
    #     ver_valid = ofm_valid.view(-1, 1).repeat(1, num_vc).view(-1) & mask_ver.view(-1)
    #     vertex_offset_kf_logits = vertex_offset_kf_logits.permute(0, 2, 3, 1).contiguous()
    #     pos_vertex_offset = vertex_offset_kf_logits[bs[ver_valid],
    #                                                 proj_vertex[ver_valid][:, 1],
    #                                                 proj_vertex[ver_valid][:, 0]].sigmoid()
    #     loss_vertex_offset = F.l1_loss(pos_vertex_offset,
    #                                    offset_vertex.view(-1, 2)[ver_valid], reduction='mean')
    #
    #     # main proj offset loss
    #     m_valid = m_mask & not_noise_mask
    #     main_offset_kf_logits = main_offset_kf_logits.permute(0, 2, 3, 1).contiguous()
    #     pos_main_offset = main_offset_kf_logits[img_id[m_valid],
    #                                             proj_main[m_valid][:, 1],
    #                                             proj_main[m_valid][:, 0]].sigmoid()
    #     loss_main_offset = F.l1_loss(pos_main_offset, offset_main[m_valid], reduction='mean')
    #
    #     loss_main_kf *= self._config.TRAINING.W_MKF
    #     loss_vertex_kf *= self._config.TRAINING.W_VKF
    #     loss_offset_fr_main *= self._config.TRAINING.W_VFM
    #     loss_main_offset *= self._config.TRAINING.W_M_OFF
    #     loss_vertex_offset *= self._config.TRAINING.W_V_OFF
    #
    #     loss = loss_main_kf + loss_vertex_kf + loss_offset_fr_main + loss_main_offset + loss_vertex_offset
    #     return loss, torch.tensor([loss_main_kf, loss_vertex_kf, loss_offset_fr_main,
    #                                loss_main_offset, loss_vertex_offset, loss],
    #                               device=loss.device).detach()

    def __call__(self, pred_logits, targets):
        # targets.to_tensor()
        # targets.apply(lambda x: x.type_as(pred_logits[0]))
        # prediction logits
        m_hm_pred = pred_logits[0]
        ver_coor_pred, m_off_pred, v_off_pred = pred_logits[1:]

        # ground truth targets
        m_hm = targets.get_field('m_hm')
        m_projs = targets.get_field('m_proj').long()
        m_offs = targets.get_field('m_off')
        # v_hm = targets.get_field('v_hm')
        ver_coor = targets.get_field('v_coor_off')
        v_projs = targets.get_field('v_proj').long()
        v_offs = targets.get_field('v_off')

        # main kf loss
        loss_main_kf = self._main_kf_loss(model_utils.sigmoid_hm(m_hm_pred), m_hm)

        # vertex kf loss
        # loss_vertex_kf = self._vertex_kf_loss(model_utils.sigmoid_hm(v_hm_pred), v_hm)
        # loss_vertex_kf = torch.zeros_like(loss_main_kf).clone().detach()

        # offset from main loss and vertex proj offset loss
        img_id = targets.get_field('img_id').long()
        m_mask = targets.get_field('mask').bool()
        not_noise_mask = targets.get_field('noise_mask').bool().bitwise_not()
        mask_3d = targets.get_field('mask_3d').bool()
        v_mask = targets.get_field('v_mask').bool()
        B, C, H, W = ver_coor_pred.shape
        num_vc = C // 2
        assert num_vc == v_projs.shape[1]
        ofm_valid = (m_mask & not_noise_mask & mask_3d)
        ofm_valid_expand = v_mask[ofm_valid].view(-1)
        # offset from main loss
        ver_coor_pred = ver_coor_pred.permute(0, 2, 3, 1).contiguous()
        ver_coor_pred = ver_coor_pred[img_id[ofm_valid],
                                      m_projs[ofm_valid][:, 1],
                                      m_projs[ofm_valid][:, 0]].view(-1, 2)
        # ver_coor_pred = (2 * ver_coor_pred.sigmoid_() - 1.) * self._ref_ver_coor.type_as(
        #     ver_coor).to(device)
        loss_ver_coor = F.l1_loss(ver_coor_pred[ofm_valid_expand],
                                  ver_coor[ofm_valid].view(-1, 2)[ofm_valid_expand], reduction='mean')

        # vertex proj offset loss
        bs = img_id.view(-1, 1).repeat(1, num_vc).view(-1)
        v_projs = v_projs.view(-1, 2)
        ver_valid = ofm_valid.view(-1, 1).repeat(1, num_vc).view(-1) & v_mask.view(-1)
        v_off_pred = v_off_pred.permute(0, 2, 3, 1).contiguous()
        pos_vertex_offset = v_off_pred[bs[ver_valid],
                                       v_projs[ver_valid][:, 1],
                                       v_projs[ver_valid][:, 0]].sigmoid()
        loss_vertex_offset = F.l1_loss(pos_vertex_offset,
                                       v_offs.view(-1, 2)[ver_valid], reduction='mean')

        # main proj offset loss
        m_valid = m_mask & not_noise_mask
        m_off_pred = m_off_pred.permute(0, 2, 3, 1).contiguous()
        pos_main_offset = m_off_pred[img_id[m_valid],
                                     m_projs[m_valid][:, 1],
                                     m_projs[m_valid][:, 0]].sigmoid()
        loss_main_offset = F.l1_loss(pos_main_offset, m_offs[m_valid], reduction='mean')

        loss_main_kf *= self._config.TRAINING.W_MKF
        # loss_vertex_kf *= self._config.TRAINING.W_VKF
        loss_ver_coor *= self._config.TRAINING.W_VFM
        loss_main_offset *= self._config.TRAINING.W_M_OFF
        loss_vertex_offset *= self._config.TRAINING.W_V_OFF

        loss = loss_main_kf + loss_ver_coor + loss_main_offset + loss_vertex_offset
        return loss, torch.tensor([loss_main_kf, loss_ver_coor,
                                   loss_main_offset, loss_vertex_offset, loss],
                                  device=loss.device).detach()

    def run(self, pred_logits, targets):
        t1 = time.time()
        # targets.to_tensor().to(pred_logits[0].device)

        # targets.apply(lambda x: x.type_as(pred_logits[0]))
        # prediction logits
        m_hm_pred = pred_logits[0]
        ver_coor_pred, m_off_pred, v_off_pred = pred_logits[1:]
        dtype = pred_logits[0].dtype
        # ground truth targets
        m_hm = targets.get_field('m_hm')
        m_projs = targets.get_field('m_proj').long()
        m_offs = targets.get_field('m_off')
        # v_hm = targets.get_field('v_hm')
        ver_coor = targets.get_field('v_coor_off')
        v_projs = targets.get_field('v_proj').long()
        v_offs = targets.get_field('v_off')

        # main kf loss
        t2 = time.time()
        loss_main_kf = self._main_kf_loss(model_utils.sigmoid_hm(m_hm_pred), m_hm)
        t3 = time.time()
        # vertex kf loss
        # loss_vertex_kf = self._vertex_kf_loss(model_utils.sigmoid_hm(v_hm_pred), v_hm)
        # loss_vertex_kf = torch.zeros_like(loss_main_kf).clone().detach()

        # offset from main loss and vertex proj offset loss
        img_id = targets.get_field('img_id').long()
        m_mask = targets.get_field('mask').bool()
        not_noise_mask = targets.get_field('noise_mask').bool().bitwise_not()
        mask_3d = targets.get_field('mask_3d').bool()
        v_mask = targets.get_field('v_mask').bool()
        B, C, H, W = ver_coor_pred.shape
        num_vc = C // 2
        assert num_vc == v_projs.shape[1]
        ofm_valid = (m_mask & not_noise_mask & mask_3d)
        ofm_valid_expand = v_mask[ofm_valid].view(-1)
        # offset from main loss
        ver_coor_pred = ver_coor_pred.permute(0, 2, 3, 1).contiguous()
        ver_coor_pred = ver_coor_pred[img_id[ofm_valid],
                                      m_projs[ofm_valid][:, 1],
                                      m_projs[ofm_valid][:, 0]].view(-1, 2)
        # ver_coor_pred = (2 * ver_coor_pred.sigmoid_() - 1.) * self._ref_ver_coor.type_as(
        #     ver_coor).to(device)
        loss_ver_coor = F.l1_loss(ver_coor_pred[ofm_valid_expand],
                                  ver_coor[ofm_valid].view(-1, 2)[ofm_valid_expand], reduction='mean')
        t4 = time.time()
        # vertex proj offset loss
        bs = img_id.view(-1, 1).repeat(1, num_vc).view(-1)
        v_projs = v_projs.view(-1, 2)
        ver_valid = ofm_valid.view(-1, 1).repeat(1, num_vc).view(-1) & v_mask.view(-1)
        v_off_pred = v_off_pred.permute(0, 2, 3, 1).contiguous()
        pos_vertex_offset = v_off_pred[bs[ver_valid],
                                       v_projs[ver_valid][:, 1],
                                       v_projs[ver_valid][:, 0]].sigmoid()
        loss_vertex_offset = F.l1_loss(pos_vertex_offset,
                                       v_offs.view(-1, 2)[ver_valid], reduction='mean')
        t5 = time.time()
        # main proj offset loss
        m_valid = m_mask & not_noise_mask
        m_off_pred = m_off_pred.permute(0, 2, 3, 1).contiguous()
        pos_main_offset = m_off_pred[img_id[m_valid],
                                     m_projs[m_valid][:, 1],
                                     m_projs[m_valid][:, 0]].sigmoid()
        loss_main_offset = F.l1_loss(pos_main_offset, m_offs[m_valid], reduction='mean')
        t6 = time.time()
        loss_main_kf *= self._config.TRAINING.W_MKF
        # loss_vertex_kf *= self._config.TRAINING.W_VKF
        loss_ver_coor *= self._config.TRAINING.W_VFM
        loss_main_offset *= self._config.TRAINING.W_M_OFF
        loss_vertex_offset *= self._config.TRAINING.W_V_OFF
        print('time:' + '%10.3g' * 5 % (t2 - t1, t3 - t2,t4 - t3,t5 - t4,t6 - t5) )
        loss = loss_main_kf + loss_ver_coor + loss_main_offset + loss_vertex_offset
        return loss, torch.tensor([loss_main_kf, loss_ver_coor,
                                   loss_main_offset, loss_vertex_offset, loss],
                                  device=loss.device).detach()

