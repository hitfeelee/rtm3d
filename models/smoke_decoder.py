import torch
import numpy as np
import torch.nn.functional as F
from datasets.data.kitti.devkit_object import utils
from utils import model_utils


class SMOKECoder(object):
    def __init__(self, config):
        self.depth_ref = config.DETECTOR.DEPTH_REF
        self.dim_ref = torch.as_tensor(config.DETECTOR.DIM_REF).to(config.DEVICE)
        self._eps = 1.e-4
        self.depth_range = (-1., 250)
        self._multi_grade_depth_def = [self.depth_ref[0], (self.depth_range[1]-3*self.depth_ref[1]-self.depth_ref[0])/2]
        self._multi_grade_depth_range = [
            [self.depth_ref[0] - 3 * self.depth_ref[1], self.depth_ref[0] + 3 * self.depth_ref[1]],
            [self.depth_ref[0] + 3 * self.depth_ref[1], self.depth_range[1]]
        ]

    def ecode_depth(self, gt_depths, depths_logits):
        '''
        Transform ground truth depth to depth offset
        '''
        gt_depth_offset = (gt_depths - self.depth_ref[0]) / self.depth_ref[1]
        _, pred_depth_offset = self.decode_depth(depths_logits)
        return gt_depth_offset, pred_depth_offset

    def decode_depth(self, depths_logits):
        '''
        Transform depth offset to depth
        '''
        sigma_max = (-self.depth_ref[0] + self.depth_range[1])/self.depth_ref[1]
        depths_offset = (2*torch.sigmoid(depths_logits) - 1) * sigma_max
        depth = depths_offset.detach() * self.depth_ref[1] + self.depth_ref[0]

        return depth, depths_offset

    def ecode_depth_multi_grade(self, gt_depth, depth_logits):
        '''
        Transform ground truth depth to depth offset
        '''
        B = len(gt_depth)
        pred_depth_conf = depth_logits[:, :2]
        pred_depth_offset = depth_logits[:, 2:]
        gt_depth_conf = torch.zeros((B, 2), dtype=torch.long).to(gt_depth.device)
        grade_0 = gt_depth <= self._multi_grade_depth_range[0][1]
        grade_1 = gt_depth > self._multi_grade_depth_range[0][1]
        gt_depth_conf[grade_0, 0] = 1
        gt_depth_conf[grade_1, 1] = 1
        gt_depth_offset = torch.zeros((B,)).type_as(gt_depth)
        gt_depth_offset[grade_0] = (gt_depth[grade_0] - self._multi_grade_depth_def[0]) / self.depth_ref[1]
        gt_depth_offset[grade_1] = (gt_depth[grade_1] - self._multi_grade_depth_def[1]) / self.depth_ref[1]
        _, pred_depth_offset = self.decode_depth_multi_grade(pred_depth_offset, gt_depth_conf)
        return gt_depth_conf, gt_depth_offset, pred_depth_conf, pred_depth_offset

    def decode_depth_multi_grade(self, depth_offset, depth_conf):
        '''
        Transform depth offset to depth
        '''
        depth_grade = torch.argmax(depth_conf, dim=1)
        B = len(depth_grade)
        sigma_max_0 = (self._multi_grade_depth_range[0][1] - self._multi_grade_depth_def[0]) / self.depth_ref[1]
        sigma_max_1 = (self._multi_grade_depth_range[1][1] - self._multi_grade_depth_def[1]) / self.depth_ref[1]
        grade_sigma = torch.tensor([[sigma_max_0, sigma_max_1]]).type_as(depth_offset).repeat([B, 1])
        grade_depth_def = torch.tensor([self._multi_grade_depth_def]).type_as(depth_offset).repeat([B, 1])
        indices = torch.arange(B).type_as(depth_grade)
        depth_offset = (2*torch.sigmoid(depth_offset[indices, depth_grade]) - 1) * grade_sigma[indices, depth_grade]
        depth = depth_offset.detach() * self.depth_ref[1] + grade_depth_def[indices, depth_grade]

        return depth, depth_offset

    def decode_location(self, centers, depths, invKs):
        '''
        retrieve objects location in camera coordinate based on projected points
        Args:
            centers: projection  of center points
            depths: object depth z
            invKs: camera intrinsic matrix, shape = [N, 3, 3]
        Returns:
            locations: objects location, shape = [N, 3]
        '''

        # number of points
        N = centers.shape[0]
        # Ks_inv = Ks.inverse()

        # transform project points in homogeneous form.
        centers_extend = torch.cat((centers, torch.ones(N, 1).type_as(centers)), dim=1)
        # expand project points as [N, 3, 1]
        centers_extend = centers_extend.unsqueeze(-1)
        # with depth
        centers_extend = centers_extend * depths.view(N, -1, 1)
        # transform image coordinates back to object locations
        locations = torch.matmul(invKs, centers_extend)

        return locations.squeeze(2)

    def ecode_dimension(self, cls_id, gt_dims, dims_logits):
        '''
        encode object dimensions to training dimension offset
        Args:
            cls_id: each object id
            gt_dims: dimension offsets, shape = (N, 3)

        Returns:

        '''
        cls_id = cls_id.flatten().long()

        dims_select = self.dim_ref[cls_id, :]
        gt_dim_offsets = torch.log(gt_dims / dims_select)
        _, pred_dim_offsets = self.decode_dimension(cls_id, dims_logits)
        return gt_dim_offsets, pred_dim_offsets

    def decode_dimension(self, cls_id, dims_logits):
        '''
        retrieve object dimensions
        Args:
            cls_id: each object id
            dims_logits: dimension logits, shape = (N, 3)

        Returns:

        '''
        cls_id = cls_id.flatten().long()
        self.dim_ref = self.dim_ref.to(cls_id.device)
        dims_select = self.dim_ref[cls_id]
        dims_offset = (2*model_utils.sigmoid_scale(dims_logits) - 0.9)
        dimensions = dims_offset.exp() * dims_select

        return dimensions, dims_offset

    def ecode_orientation(self, vector_ori, gt_alpha):
        pass

    def decode_orientation(self, vector_ori, locations):
        '''
        retrieve object orientation
        Args:
            vector_ori: local orientation in [sin, cos] format
            locations: object location

        Returns: for training we only need roty
                 for testing we need both alpha and roty

        '''

        locations = locations.view(-1, 3)
        rays = torch.atan2(locations[:, 0], locations[:, 2])
        alphas = torch.atan2(vector_ori[:, 0], vector_ori[:, 1])

        # retrieve object rotation y angle.
        rotys = alphas + rays

        # in training time, it does not matter if angle lies in [-PI, PI]
        # it matters at inference time? todo: does it really matter if it exceeds.
        larger_idx = (rotys > np.pi).nonzero(as_tuple=True)[0]
        small_idx = (rotys < -np.pi).nonzero(as_tuple=True)[0]

        if len(larger_idx) != 0:
            rotys[larger_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            rotys[small_idx] += 2 * np.pi

        return rotys, alphas

    def encode_smoke_pred3d_and_targets(self, pred3d_logits, targets):
        '''

        :param pred3d_logits: (B, K, H, W), here K is (z_off, sin(alpha), cos(alpha), h, w, l)
        :param targets:
        :return:
        '''
        m_mask = targets.get_field('mask').clone().bool()
        not_noise_mask = targets.get_field('noise_mask').clone().bool().bitwise_not()
        m_valid = m_mask & not_noise_mask
        clses = targets.get_field('class').clone().long()[m_valid]
        m_projs = targets.get_field('m_proj').clone().long()[m_valid]
        img_id = targets.get_field('img_id').clone().long()[m_valid]
        locations = targets.get_field('location').clone()[m_valid]
        alphas = targets.get_field('alpha').clone()[m_valid]
        dimensions = targets.get_field('dimension').clone()[m_valid]
        C = pred3d_logits.shape[1]
        pred3d_logits = pred3d_logits.permute(0, 2, 3, 1).contiguous()
        pred3d_logits = pred3d_logits[img_id,
                                      m_projs[:, 1],
                                      m_projs[:, 0]].view(-1, C)

        pred_orients = F.normalize(pred3d_logits[:, -5:-3], p=2, dim=-1)
        gt_dims, pred_dims = self.ecode_dimension(clses, dimensions, pred3d_logits[:, -3:])
        gt_depths, pred_depths = self.ecode_depth(locations[:, 2], pred3d_logits[:, 0])
        return pred_dims, pred_depths, pred_orients, gt_dims, gt_depths, alphas

    def encode_smoke_pred3d_and_targets_multi_grade(self, pred3d_logits, targets):
        '''

        :param pred3d_logits: (B, K, H, W), here K is (z_off, sin(alpha), cos(alpha), h, w, l)
        :param targets:
        :return:
        '''
        m_mask = targets.get_field('mask').clone().bool()
        not_noise_mask = targets.get_field('noise_mask').clone().bool().bitwise_not()
        m_valid = m_mask & not_noise_mask
        clses = targets.get_field('class').clone().long()[m_valid]
        m_projs = targets.get_field('m_proj').clone().long()[m_valid]
        img_id = targets.get_field('img_id').clone().long()[m_valid]
        locations = targets.get_field('location').clone()[m_valid]
        alphas = targets.get_field('alpha').clone()[m_valid]
        dimensions = targets.get_field('dimension').clone()[m_valid]
        C = pred3d_logits.shape[1]
        pred3d_logits = pred3d_logits.permute(0, 2, 3, 1).contiguous()
        pred3d_logits = pred3d_logits[img_id,
                                      m_projs[:, 1],
                                      m_projs[:, 0]].view(-1, C)

        pred_orients = F.normalize(pred3d_logits[:, -5:-3], p=2, dim=-1)
        gt_dims, pred_dims = self.ecode_dimension(clses, dimensions, pred3d_logits[:, -3:])
        gt_depth_confs, gt_depth_offsets, pred_depth_confs, pred_depth_offsets = self.ecode_depth_multi_grade(
            locations[:, 2], pred3d_logits[:, :4])
        return pred_dims, (pred_depth_confs, pred_depth_offsets), pred_orients, \
               gt_dims, (gt_depth_confs, gt_depth_offsets), alphas

    def decode_smoke_pred3d_logits(self, img_id, centers, centers_off, clses, pred3d_logits, invKs):
        '''

        :param img_id:
        :param centers:
        :param centers_off:
        :param clses:
        :param pred3d_logits: (B, K, H, W), here K is (z_off, sin(alpha), cos(alpha), h, w, l)
        :param invKs: (B, 3, 3)
        :return:
        '''
        K = pred3d_logits.shape[1]
        pred3d_logits = pred3d_logits.permute(0, 2, 3, 1).contiguous()
        pred3d_logits = pred3d_logits[img_id,
                                      centers[:, 1].long(),
                                      centers[:, 0].long()].view(-1, K)

        pred_dims = self.decode_dimension(clses, pred3d_logits[:, -3:])[0]
        pred_depths = self.decode_depth(pred3d_logits[:, 0])[0]
        # pred_depths = self.decode_depth_multi_grade(pred3d_logits[:, 2:4], pred3d_logits[:, :2])[0]
        pred_locs = self.decode_location(centers + centers_off,
                                         pred_depths, invKs)
        pred_rys, pred_alpha = self.decode_orientation(F.normalize(pred3d_logits[:, -5:-3], p=2, dim=-1), pred_locs)
        return pred_dims, pred_locs, pred_rys, pred_alpha
