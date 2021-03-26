import torch.nn as nn
from models.nets.keypoint_fpn_fusion import KeypointFPNFusion
from models.nets.header import RTM3DHeader
import torch
from utils import model_utils
from utils import torch_utils
from models.smoke_decoder import SMOKECoder
import time


class Model(nn.Module):
    def __init__(self, config, backbone):
        super(Model, self).__init__()
        self.config = config
        self.backbone = backbone
        self.kfpn_fusion = KeypointFPNFusion(config, self.backbone.kfpn_spec)
        self.detect_header = RTM3DHeader(config)
        self.smoke_encoder = SMOKECoder(config)
        self.export = False
        torch_utils.initialize_weights(self)

    def forward(self, x, invKs=None):
        x = self.backbone(x)
        x = self.kfpn_fusion(x)

        pred_logits = self.detect_header(x)
        if self.training or self.export:
            return pred_logits
        else:
            return self.inference([p.clone() for p in pred_logits], invKs), pred_logits

    def inference(self, pred_logits, invKs):
        main_kf_logits = pred_logits[0]
        regress_logits = pred_logits[1]
        Bs = main_kf_logits.shape[0]
        results = [None] * Bs
        for i in range(Bs):  # process per image
            clses_i, m_scores_i, m_projs_i = self._obtain_main_proj2d(main_kf_logits[i],
                                                                      self.config.DETECTOR.SCORE_THRESH,
                                                                      self.config.DETECTOR.TOPK_CANDIDATES)

            if len(clses_i) == 0 or invKs[i] is None:
                continue
            K_i = invKs[i].repeat(len(clses_i), 1, 1)
            m_projs_offset_i = regress_logits[i][:2, m_projs_i[1].long(), m_projs_i[0].long()].sigmoid_()
            m_projs_i = torch.cat([m_projs_i[0].unsqueeze(-1), m_projs_i[1].unsqueeze(-1)], dim=-1)
            m_projs_offset_i = torch.cat([m_projs_offset_i[0].unsqueeze(-1), m_projs_offset_i[1].unsqueeze(-1)], dim=-1)
            pred_dims_i, pred_locs_i, pred_rys_i, pred_alpha_i = self.smoke_encoder.decode_smoke_pred3d_logits(
                i, m_projs_i, m_projs_offset_i, clses_i, regress_logits[:, 2:, :, :], K_i)

            m_projs_i += m_projs_offset_i

            # m_projs_i *= self.config.MODEL.DOWN_SAMPLE  # (N, 2)

            result = torch.cat([
                clses_i.view(-1, 1).type_as(pred_alpha_i), pred_alpha_i.view(-1, 1), pred_dims_i, pred_locs_i,
                pred_rys_i.view(-1, 1), m_scores_i.view(-1, 1)
            ], dim=1)
            results[i] = result

        return results

    def fuse(model):
        for m in model.modules():
            print(type(m))
            if type(m) == RTM3DHeader:
                m.fuse()
            # if type(m) is Conv:
            #     m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
            #     m.bn = None  # remove batchnorm
            #     m.forward = m.fuseforward  # update forward

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
        return cls, scores, [x.type_as(main_kf_logits), y.type_as(main_kf_logits)]

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

        return offset_fr_main


