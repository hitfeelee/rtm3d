import torch
import numpy as np
import torch.nn.functional as F
from datasets.data.kitti.devkit_object import utils


class SMOKECoder(object):
    def __init__(self, config, device="cuda"):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.depth_ref = config.DETECTOR.DEPTH_REF
        self.dim_ref = torch.as_tensor(config.DETECTOR.DIM_REF).to(device=device)

    def ecode_depth(self, gt_depths):
        '''
        Transform ground truth depth to depth offset
        '''
        depth_offset = (gt_depths - self.depth_ref[0]) / self.depth_ref[1]
        return depth_offset

    def decode_depth(self, depths_offset):
        '''
        Transform depth offset to depth
        '''
        sigma_min = (-self.depth_ref[0] + 1e-6)/self.depth_ref[1]
        sigma_max = (-self.depth_ref[0] + 200)/self.depth_ref[1]
        depths_offset = depths_offset.sigmoid_() * (sigma_max - sigma_min) + sigma_min
        depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]

        return depth

    def decode_location(self, centers, depths, Ks):
        '''
        retrieve objects location in camera coordinate based on projected points
        Args:
            centers: projection  of center points
            depths: object depth z
            Ks: camera intrinsic matrix, shape = [N, 3, 3]
        Returns:
            locations: objects location, shape = [N, 3]
        '''
        device = centers.device

        # number of points
        N = centers.shape[0]
        Ks_inv = Ks.inverse()

        # transform project points in homogeneous form.
        centers_extend = torch.cat((centers, torch.ones(N, 1).to(device=device)), dim=1)
        # expand project points as [N, 3, 1]
        centers_extend = centers_extend.unsqueeze(-1)
        # with depth
        centers_extend = centers_extend * depths.view(N, -1, 1)
        # transform image coordinates back to object locations
        locations = torch.matmul(Ks_inv, centers_extend)

        return locations.squeeze(2)

    def ecode_dimension(self, cls_id, gt_dims):
        '''
        encode object dimensions to training dimension offset
        Args:
            cls_id: each object id
            gt_dims: dimension offsets, shape = (N, 3)

        Returns:

        '''
        cls_id = cls_id.flatten().long()

        dims_select = self.dim_ref[cls_id, :]
        dim_offsets = torch.log(gt_dims / dims_select)

        return dim_offsets

    def decode_dimension(self, cls_id, dims_offset):
        '''
        retrieve object dimensions
        Args:
            cls_id: each object id
            dims_offset: dimension offsets, shape = (N, 3)

        Returns:

        '''
        cls_id = cls_id.flatten().long()
        self.dim_ref = self.dim_ref.to(cls_id.device)
        dims_select = self.dim_ref[cls_id]
        dimensions = (dims_offset.sigmoid_() - 0.55).exp() * dims_select

        return dimensions

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

        pred_dims = pred3d_logits[:, -3:].sigmoid() - 0.55
        sigma_min = (-self.depth_ref[0] + 1e-6) / self.depth_ref[1]
        sigma_max = (-self.depth_ref[0] + 200) / self.depth_ref[1]
        pred_depths = pred3d_logits[:, 0].sigmoid() * (sigma_max - sigma_min) + sigma_min
        pred_orients = F.normalize(pred3d_logits[:, 1:3], p=2, dim=-1)
        gt_dims = self.ecode_dimension(clses, dimensions)
        gt_depths = self.ecode_depth(locations[:, 2])
        return pred_dims, pred_depths, pred_orients, gt_dims, gt_depths, alphas

    def decode_smoke_pred3d_logits_train(self, pred3d_logits, targets):
        '''

        :param pred3d_logits: (B, K, H, W), here K is (z_off, sin(alpha), cos(alpha), h, w, l)
        :param targets:
        :return:
        '''
        clses = targets.get_field('class').long()
        m_projs = targets.get_field('m_proj').long()
        m_offs = targets.get_field('m_off')
        img_id = targets.get_field('img_id').long()
        m_mask = targets.get_field('mask').bool()
        not_noise_mask = targets.get_field('noise_mask').bool().bitwise_not()
        locations = targets.get_field('location')
        Rys = targets.get_field('Ry')
        dimensions = targets.get_field('dimension')
        Ks = targets.get_field('K')
        m_valid = m_mask & not_noise_mask
        pred_dims, pred_locs, pred_rys, _ = self.decode_smoke_pred3d_logits_eval(
            img_id[m_valid], m_projs[m_valid], m_offs[m_valid], clses[m_valid], pred3d_logits, Ks[m_valid]
        )
        pred_vertexs_dim, _, _ = utils.calc_proj2d_bbox3d(pred_dims.detach().cpu().numpy(),
                                                          locations[m_valid].detach().cpu().numpy(),
                                                          Rys[m_valid].detach().cpu().numpy(),
                                                          Ks[m_valid].detach().cpu().numpy().reshape(-1, 3, 3))

        pred_vertexs_loc, _, _ = utils.calc_proj2d_bbox3d(dimensions[m_valid].detach().cpu().numpy(),
                                                          pred_locs.detach().cpu().numpy(),
                                                          Rys[m_valid].detach().cpu().numpy(),
                                                          Ks[m_valid].detach().cpu().numpy().reshape(-1, 3, 3))

        pred_vertexs_ry, _, _ = utils.calc_proj2d_bbox3d(dimensions[m_valid].detach().cpu().numpy(),
                                                         locations[m_valid].detach().cpu().numpy(),
                                                         pred_rys.detach().cpu().numpy(),
                                                         Ks[m_valid].detach().cpu().numpy().reshape(-1, 3, 3))
        pred_vertexs_dim = torch.from_numpy(pred_vertexs_dim[..., :-1]
                                            ).type_as(pred3d_logits).permute(0, 2, 1).contiguous()
        pred_vertexs_loc = torch.from_numpy(pred_vertexs_loc[..., :-1]
                                            ).type_as(pred3d_logits).permute(0, 2, 1).contiguous()
        pred_vertexs_ry = torch.from_numpy(pred_vertexs_ry[..., :-1]
                                            ).type_as(pred3d_logits).permute(0, 2, 1).contiguous()
        return pred_vertexs_dim, pred_vertexs_loc, pred_vertexs_ry

    def decode_smoke_pred3d_logits_eval(self, img_id, centers, centers_off, clses, pred3d_logits, Ks):
        '''

        :param img_id:
        :param centers:
        :param centers_off:
        :param clses:
        :param pred3d_logits: (B, K, H, W), here K is (z_off, sin(alpha), cos(alpha), h, w, l)
        :param Ks:
        :return:
        '''
        K = pred3d_logits.shape[1]
        pred3d_logits = pred3d_logits.permute(0, 2, 3, 1).contiguous()
        pred3d_logits = pred3d_logits[img_id,
                                      centers[:, 1],
                                      centers[:, 0]].view(-1, K)

        pred_dims = self.decode_dimension(clses, pred3d_logits[:, -3:])
        pred_depths = self.decode_depth(pred3d_logits[:, 0])
        pred_locs = self.decode_location(centers.float() + centers_off,
                                         pred_depths, Ks.view(-1, 3, 3))
        pred_rys, pred_alpha = self.decode_orientation(F.normalize(pred3d_logits[:, 1:3], p=2, dim=-1), pred_locs)
        return pred_dims, pred_locs, pred_rys, pred_alpha
