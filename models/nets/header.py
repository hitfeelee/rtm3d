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
        _dilation = [6] + [1] * (_num_conv - 1)
        # main detect head
        self.main_kf_header = torch_utils.make_conv_level(_in_ch, _in_ch, 3, _num_conv, bias=True,
                                                          dilation=_dilation)
        self.main_kf_header.add_module('main_kf_head', nn.Conv2d(_in_ch, _num_class, 3, padding=1, bias=True))

        # vertex offset from main kf
        self.offset_fr_main_header = torch_utils.make_conv_level(_in_ch, _in_ch, 3, _num_conv, bias=True,
                                                                 dilation=_dilation)
        self.offset_fr_main_header.add_module('offset_fr_main_head', nn.Conv2d(_in_ch, 16, 3, padding=1, bias=True))

        # main kf offset
        self.main_offset_header = torch_utils.make_conv_level(_in_ch, _in_ch, 3, _num_conv, bias=True,
                                                         dilation=_dilation)
        self.main_offset_header.add_module('main_offset_head', nn.Conv2d(_in_ch, 2, 3, padding=1, bias=True))

        # 3d properties # (z_off, sin(alpha), cos(alpha), h, w, l)
        self.pred3d_header = torch_utils.make_conv_level(_in_ch, _in_ch, 3, _num_conv, bias=True,
                                                         dilation=_dilation)
        self.pred3d_header.add_module('pred3d_head', nn.Conv2d(_in_ch, 6, 3, padding=1, bias=True))

    def forward(self, x):
        main_kf_logits = self.main_kf_header(x)
        offset_fr_main_logits = self.offset_fr_main_header(x)
        main_offset_kf_logits = self.main_offset_header(x)
        pred3d_logits = self.pred3d_header(x)
        return main_kf_logits, offset_fr_main_logits, main_offset_kf_logits, pred3d_logits
