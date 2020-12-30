import torch
import torch.nn as nn
import math
from models.nets.module import *


class KeypointFPNFusion(nn.Module):
    def __init__(self, config, kfpn_spec):
        super(KeypointFPNFusion, self).__init__()
        self.config = config
        _kfns = self.config.MODEL.KFNs
        _kfns_strides = [kfpn_spec[k].stride for k in _kfns]
        _kfns_channels = [kfpn_spec[k].channels for k in _kfns]
        _out_channels = self.config.MODEL.OUT_CHANNELS
        assert all((_kfns_strides[i] / _kfns_strides[i - 1]) == 2 for i in range(1, len(_kfns_strides)))
        _kfpn_levels = [int(math.log2(s)) for s in _kfns_strides]
        self._kfpn_levels = _kfpn_levels
        for i in range(len(_kfpn_levels) - 1, 0, -1):
            head = nn.Conv2d(_kfns_channels[i], _out_channels, 1, 1, bias=True)
            setattr(self, 'kfpn_head{}'.format(_kfpn_levels[i]), head)
            up = UpSample(_out_channels)
            setattr(self, 'kfpn_up{}'.format(_kfpn_levels[i]), up)
            proj = nn.Conv2d(_kfns_channels[i - 1] + _out_channels, _kfns_channels[i - 1], 1, 1, bias=True)
            setattr(self, 'kfpn_proj{}'.format(_kfpn_levels[i]), proj)

        head = nn.Conv2d(_kfns_channels[0], _out_channels, 1, 1, bias=True)
        setattr(self, 'kfpn_head{}'.format(_kfpn_levels[0]), head)

        _fusion_levels = [int(l - _kfpn_levels[0]) for l in _kfpn_levels]

        for i in range(len(_fusion_levels) - 1, 0, -1):
            up = nn.Sequential(*[UpSample(_out_channels) for _ in range(_fusion_levels[i])])
            setattr(self, 'fusion_up{}'.format(_kfpn_levels[i]), up)

    def _fpn(self, x):
        assert len(x) == len(self._kfpn_levels)
        for i in range(len(self._kfpn_levels) - 1, 0, -1):
            head = getattr(self, 'kfpn_head{}'.format(self._kfpn_levels[i]))
            up = getattr(self, 'kfpn_up{}'.format(self._kfpn_levels[i]))
            proj = getattr(self, 'kfpn_proj{}'.format(self._kfpn_levels[i]))
            x[i] = head(x[i])
            x[i - 1] = proj(torch.cat([up(x[i]), x[i - 1]], dim=1))

        head = getattr(self, 'kfpn_head{}'.format(self._kfpn_levels[0]))
        x[0] = head(x[0])
        return x

    def forward(self, x):
        # assert len(x) == len(self._kfpn_levels)
        # y = [None] * len(x)
        # for i in range(len(self._kfpn_levels) - 1, 0, -1):
        #     head = getattr(self, 'kfpn_head{}'.format(self._kfpn_levels[i]))
        #     up = getattr(self, 'kfpn_up{}'.format(self._kfpn_levels[i]))
        #     proj = getattr(self, 'kfpn_proj{}'.format(self._kfpn_levels[i]))
        #     y[i] = head(x[i])
        #     x[i - 1] = proj(torch.cat([up(y[i]), x[i - 1]], dim=1))
        #
        # head = getattr(self, 'kfpn_head{}'.format(self._kfpn_levels[0]))
        # y[0] = head(x[0])
        out = self._fpn(x)
        # out = [f.detach() for f in y]
        z = out[0]
        for i in range(len(self._kfpn_levels) - 1, 0, -1):
            up = getattr(self, 'fusion_up{}'.format(self._kfpn_levels[i]))
            out_i_up = up(out[i])
            bs, c, h, w = out_i_up.shape
            z_i = out_i_up * torch.softmax(out_i_up.detach().view(bs, c, -1), dim=-1).view(bs, c, h, w)
            z += z_i
        return z








