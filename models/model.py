import torch.nn as nn
from models.nets.keypoint_fpn_fusion import KeypointFPNFusion
from models.nets.header import RTM3DHeader
import torch
from utils import model_utils
from utils import torch_utils


class Model(nn.Module):
    def __init__(self, config, backbone):
        super(Model, self).__init__()
        self.config = config
        self.backbone = backbone
        self.kfpn_fusion = KeypointFPNFusion(config, self.backbone.kfpn_spec)
        self.detect_header = RTM3DHeader(config)
        self.export = False
        self._ref_offset_fr_main = torch.tensor([self.config.DATASET.VERTEX_OFFSET_INFER])
        torch_utils.initialize_weights(self)

    def forward(self, x):
        x = self.backbone(x)
        x = self.kfpn_fusion(x)
        pred_logits = self.detect_header(x)
        if self.training:
            return pred_logits
        else:
            return self.inference([p.clone() for p in pred_logits]), pred_logits

    def inference(self, pred_logits):
        main_kf_logits = pred_logits[0]
        offset_fr_main_logits, main_offset_kf_logits, vertex_offset_kf_logits = pred_logits[1:]
        Bs = main_kf_logits.shape[0]
        clses = [None] * Bs
        m_scores = [None] * Bs
        m_projs = [None] * Bs
        v_scores = [None] * Bs
        v_projs = [None] * Bs
        v_projs_regress = [None] * Bs
        bboxes_2d = [None] * Bs
        for i in range(Bs):  # process per image
            clses_i, m_scores_i, m_projs_i = self._obtain_main_proj2d(main_kf_logits[i], self.config.DETECTOR.SCORE_THRESH,
                                                                self.config.DETECTOR.TOPK_CANDIDATES)
            if len(clses_i) == 0:
                continue
            # v_scores_i, v_projs_i = self._obtain_vertex_proj2d(vertex_kf_logits[i],
            #                                                self.config.DETECTOR.TOPK_CANDIDATES)
            offsets_fr_main_i = self._obtain_offset_fr_main(offset_fr_main_logits[i], m_projs_i)
            m_projs_offset_i = main_offset_kf_logits[i][:, m_projs_i[1].long(), m_projs_i[0].long()].sigmoid_()
            m_projs_i[0] += m_projs_offset_i[0]
            m_projs_i[1] += m_projs_offset_i[1]
            # (K , topK) -> (K x topK, )
            # v_projs_i[0] = v_projs_i[0].view(-1)
            # v_projs_i[1] = v_projs_i[1].view(-1)

            # v_projs_offset_i = vertex_offset_kf_logits[i][:, v_projs_i[1].long(), v_projs_i[0].long()].sigmoid_()
            # v_projs_i[0] += v_projs_offset_i[0]
            # v_projs_i[1] += v_projs_offset_i[1]
            # (K x topK, ) -> (K , topK)
            # v_projs_i[0] = v_projs_i[0].view(-1, self.config.DETECTOR.TOPK_CANDIDATES)
            # v_projs_i[1] = v_projs_i[1].view(-1, self.config.DETECTOR.TOPK_CANDIDATES)
            # v_projs_i, v_projs_regress_i, v_scores_i = self._group_vertexs_kf(m_projs_i, v_projs_i,
            #                                                                   v_scores_i, offsets_fr_main_i)
            m_projs_i = torch.cat([m_projs_i[0].unsqueeze(-1), m_projs_i[1].unsqueeze(-1)], dim=-1)
            v_projs_regress_i = offsets_fr_main_i.permute(1, 0, 2).contiguous() + m_projs_i.view(-1, 1, 2)
            clses[i] = clses_i  # (N,)
            m_scores[i] = m_scores_i  # (N,)
            m_projs[i] = self.config.MODEL.DOWN_SAMPLE * m_projs_i  # (N, 2)
            # v_scores[i] = v_scores_i  # (N, 8)
            # v_projs[i] = self.config.MODEL.DOWN_SAMPLE * v_projs_i  # (N, 8, 2)
            v_projs_regress[i] = self.config.MODEL.DOWN_SAMPLE * v_projs_regress_i  # (N, 8, 2)
            oc_min = torch.min(v_projs_regress[i], dim=1)[0]
            oc_max = torch.max(v_projs_regress[i], dim=1)[0]
            bboxes_2d[i] = torch.cat([oc_min, oc_max], dim=-1)

        return clses, m_scores, m_projs, v_projs_regress, bboxes_2d

    def _obtain_main_proj2d(self, main_kf_logits, confidence, topK=30):
        '''
        select main proj points of the bboxs of objects from heatmap
        :param main_kf_logits: shape->(K, H, W) , K is the num of the classes
        :param confidence: the min confidence for selecting
        :param topK: the max num of objects
        :return: proj points
        '''
        main_hm = main_kf_logits.sigmoid_()
        main_hm = model_utils.nms_hm(main_hm.unsqueeze(dim=0), 3).squeeze(dim=0)
        K, H, W = main_hm.shape
        # (K, H, W) -> (KxHxW)
        main_hm = main_hm.view(-1)
        scores, indices = torch.topk(main_hm, topK, dim=0)
        keep = scores > confidence
        scores = scores[keep]
        indices = indices[keep]
        cls = indices // (H * W)
        xy = indices % (H * W)
        y = xy // W
        x = xy % W
        return cls, scores, [x.to(torch.float32), y.to(torch.float32)]

    def _obtain_vertex_proj2d(self, vertex_kf_logits, topK=30):
        '''
        select main proj points of the bboxs of objects from heatmap
        :param vertex_kf_logits: shape->(K, H, W) , K is the num of the vertex
        :param topK: the max num of objects
        :return: proj points, shape->(K, topK)
        '''
        vertex_hm = vertex_kf_logits.sigmoid_()
        vertex_hm = model_utils.nms_hm(vertex_hm.unsqueeze(dim=0), 3).squeeze(dim=0)
        K, H, W = vertex_hm.shape
        # (K, H, W) -> (K, HxW)
        vertex_hm = vertex_hm.view(K, -1)
        scores, indices = torch.topk(vertex_hm, topK, dim=-1)
        y = indices // W
        x = indices % W
        return scores, [x.to(torch.float32), y.to(torch.float32)]

    def _obtain_offset_fr_main(self, offset_fr_main_logits, main_projs):
        '''
        obtain teh offsets of vertexs from main point
        :param offset_fr_main_logits: shape->(K, H, W), here K -> 16
        :param main_projs: shape->(K, N), here K -> 8
        :return: shape -> (K, N, 2)
        '''
        _, H, W = offset_fr_main_logits.shape
        x_projs, y_projs = main_projs
        N = len(x_projs)
        offset_fr_main = offset_fr_main_logits[:, y_projs.long(), x_projs.long()].view(
            -1, 2, N).permute(0, 2, 1).contiguous()

        # offset_fr_main = (2 * offset_fr_main.sigmoid_() - 1)
        # gain = torch.tensor([[W, H]]).type_as(offset_fr_main).to(offset_fr_main.device)
        return offset_fr_main  # * (self._ref_offset_fr_main.to(offset_fr_main.device) * gain)

    def _group_vertexs_kf(self, m_projs, v_projs, v_scores, offsets_fr_main):
        '''

        :param m_projs: main kf points (x, y) , (shape (N,), shape (N,))
        :param v_projs: vertex kf points of bbox3d, (shape (K, topK), shape (K, topK))
        :param v_scores: the scores of vertex kf points, shape (K, topK)
        :param offsets_fr_main: offsets of vertex from main kf, (K, N, 2)
        :return:
        '''

        m_projs = torch.cat([m_projs[0].unsqueeze(-1), m_projs[1].unsqueeze(-1)], dim=-1)
        v_projs = torch.cat([v_projs[0].unsqueeze(-1), v_projs[1].unsqueeze(-1)], dim=-1)
        K, N, _ = offsets_fr_main.shape
        offsets_kf = v_projs.unsqueeze(dim=1) - m_projs.view(1, -1, 1, 2)  # (K, N, topK, 2)

        diff_xy = offsets_kf - offsets_fr_main.unsqueeze(dim=-2)
        diff_dis = diff_xy[..., 0] ** 2 + diff_xy[..., 1] ** 2  # (K, N, topK)
        indices = torch.argmin(diff_dis, dim=-1)
        i_K = torch.arange(0, K).to(m_projs.device).long()
        i_N = torch.arange(0, N).to(m_projs.device).long()
        i_K, i_N = torch.meshgrid([i_K, i_N])
        v_projs_kf = v_projs.unsqueeze(dim=1).repeat(1, N, 1, 1
                                                        )[i_K.flatten(), i_N.flatten(), indices.flatten()
        ].view(K, N, 2).permute(1, 0, 2).contiguous()
        v_projs_regressed = offsets_fr_main.permute(1, 0, 2).contiguous() + m_projs.view(-1, 1, 2)
        v_scores_kf = v_scores.unsqueeze(dim=1).repeat(1, N, 1
                                                       )[i_K.flatten(), i_N.flatten(), indices.flatten()
        ].view(K, N).permute(1, 0).contiguous()
        return v_projs_kf, v_projs_regressed, v_scores_kf


