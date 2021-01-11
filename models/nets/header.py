import torch
import torch.nn as nn
from utils import torch_utils


class RTM3DHeader(nn.Module):
    def __init__(self, config):
        super(RTM3DHeader, self).__init__()
        self._config = config
        _in_ch = self._config.MODEL.OUT_CHANNELS
        _num_class = len(self._config.DATASET.OBJs)
        _num_conv = self._config.MODEL.HEADER_NUM_CONV
        _dilation = [1] + [1] * (_num_conv - 1)
        # main detect head
        self.main_kf_header = torch_utils.make_conv_level(_in_ch, _in_ch, 3, _num_conv, bias=True,
                                                          dilation=_dilation)
        self.main_kf_header.add_module('main_kf_head', nn.Conv2d(_in_ch, _num_class, 3, padding=1, bias=True))

        # 3d properties # (z_off, sin(alpha), cos(alpha), h, w, l)
        self.regress_header = torch_utils.make_conv_level(_in_ch, _in_ch, 3, _num_conv, bias=True,
                                                         dilation=_dilation)
        self.regress_header.add_module('regress_head', nn.Conv2d(_in_ch, 8, 3, padding=1, bias=True))

    def forward(self, x):
        main_kf_logits = self.main_kf_header(x)
        regress_logits = self.regress_header(x)
        return main_kf_logits, regress_logits

    def fuse(self):
        self.main_kf_header = torch_utils.fuse_conv_and_bn_in_sequential(self.main_kf_header)
        self.regress_header = torch_utils.fuse_conv_and_bn_in_sequential(self.regress_header)
