import torch.nn as nn
import torch
import copy
import math


class UpSample(nn.Module):
    def __init__(self, c1, c2=None, k=2):
        super(UpSample, self).__init__()
        # o = (i - 1) * s + k - 2*p + op
        c2 = c2 if c2 is not None else c1
        self.conv_tran = nn.ConvTranspose2d(c1, c2, k*2, stride=k, padding=k // 2, output_padding=0, bias=False)

    def forward(self, x):
        return self.conv_tran(x)


class FocalLoss1(nn.Module):
    def __init__(self, alpha=2., beda=4.):
        '''

        :param alpha:
        :param beda:
        '''
        super(FocalLoss1, self).__init__()
        self._alpha = alpha
        self._beda = beda

    def forward(self, prediction, targets):
        pos = (targets == 1)
        neg = (targets != 1)
        loss = torch.zeros_like(prediction)

        loss[pos] = torch.pow((1 - prediction[pos]), self._alpha) * torch.log(prediction[pos])
        loss[neg] = torch.pow((1 - targets[neg]), self._beda) * prediction[neg] * torch.log(1 - prediction[neg])
        num_pos = max(1, pos.float().sum())
        loss = -1 * loss.sum() / num_pos
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.

        positive_loss = torch.log(prediction) \
                        * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) \
                        * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss -= negative_loss
        else:
            loss -= (positive_loss + negative_loss) / num_positive

        return loss


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """

    def __init__(self, model, decay=0.9999, device=''):
        # make a copy of the model for accumulating moving average of weights
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.updates = 0  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.device = device  # perform model on different device from model if set
        if device:
            self.model.to(device=device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
                msd, esd = model.module.state_dict(), self.model.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.model.state_dict()

            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model):
        # Assign attributes (which may change during training)
        for k in model.__dict__.keys():
            if not k.startswith('_'):
                setattr(self.model, k, getattr(model, k))
