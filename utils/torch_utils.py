import math
import os
import time
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.distributed as dist
import threading
from models.nets.module import ConvBn

hinf = 65504.

def init_seeds(seed=0):
    torch.manual_seed(seed)

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def select_device(device='', apex=False, batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def _fill_up_weights(up):
    # todo: we can replace math here?
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.ConvTranspose2d:
            _fill_up_weights(m)
        elif t is nn.BatchNorm2d:
            m.eps = 1e-4
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device)
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        fusedconv = fusedconv.to(conv.weight.device,conv.weight.dtype)
        return fusedconv


def fuse_conv_and_bn_in_sequential(sq):
    modules = []
    if type(sq) == nn.Sequential:
        for i in range(len(sq) - 1):
            if type(sq[i]) == nn.Conv2d and type(sq[i+1]) == nn.BatchNorm2d:
                modules.append(fuse_conv_and_bn(sq[i], sq[i+1]))
            elif type(sq[i]) == nn.BatchNorm2d:
                continue
            else:
                modules.append(sq[i])
        return nn.Sequential(*modules)
    else:
        return sq


def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 480, 640),), verbose=False)
        fs = ', %.1f GFLOPS' % (macs / 1E9 * 2)
    except:
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = models.__dict__[name](pretrained=True)

    # Display model properties
    input_size = [3, 224, 224]
    input_space = 'RGB'
    input_range = [0, 1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for x in [input_size, input_space, input_range, mean, std]:
        print(x + ' =', eval(x))

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = torch.nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = torch.nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False):  # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        gs = 32  # (pixels) grid size
        h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def normalize(input, p=2, dim=-1, eps=1e-12):
    norm = torch.norm(input, p=p, dim=dim, keepdim=True)
    clip = norm < eps
    norm[clip] = eps
    input = input / norm
    return input


def make_conv_level(in_channels, out_channels, kernel_size=3, num_convs=1, norm_func=nn.BatchNorm2d,
                     stride=1, dilation=1, bias=False):
    """
    make conv layers based on its number.
    """
    modules = []
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * num_convs
    if isinstance(out_channels, int):
        out_channels = [in_channels] * (num_convs - 1) + [out_channels]
    if isinstance(norm_func, int):
        norm_func = [norm_func] * num_convs
    if isinstance(dilation, int):
        dilation = [dilation] * num_convs

    for i in range(num_convs):
        s = stride if i == 0 else 1
        padding = (kernel_size[i] - 1) * dilation[i] // 2
        modules.extend([
            nn.Conv2d(in_channels, out_channels[i], kernel_size=kernel_size[i],
                      stride=s,
                      padding=padding, bias=bias, dilation=dilation[i]),
            norm_func(out_channels[i]),
            nn.ReLU(inplace=True)])

    return nn.Sequential(*modules)


def make_convbn_level(in_channels, out_channels, kernel_size=3, num_convs=1,
                     stride=1, dilation=1, bias=False):
    """
    make conv layers based on its number.
    """
    modules = []
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * num_convs
    if isinstance(out_channels, int):
        out_channels = [in_channels] * (num_convs - 1) + [out_channels]
    if isinstance(dilation, int):
        dilation = [dilation] * num_convs

    for i in range(num_convs):
        s = stride if i == 0 else 1
        padding = (kernel_size[i] - 1) * dilation[i] // 2
        modules.extend([
            ConvBn(in_channels, out_channels[i], kernel_size=kernel_size[i],
                   stride=s, padding=padding, bias=bias, dilation=dilation[i]),
            nn.ReLU(inplace=True)])

    return nn.Sequential(*modules)


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


class MultiThread (threading.Thread):
    def __init__(self, threadID, callback, args):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.callback = callback
        self.args = args
        self._result = None

    def run(self):
        # print("开始线程：" + str(self.threadID))
        self._result = self.callback(*self.args)
        # print("退出线程：" + str(self.threadID))

    def get_result(self):
        return self._result


def multi_thread_to_device(tensor, device, nw=4):

    def worker(sub_tensor, device):
        return sub_tensor.to(device)

    N = math.ceil(len(tensor) / nw)
    th_pools = []
    for i in range(nw):
        start = i * N
        end = min((i + 1) * N, len(tensor))
        th = MultiThread(i, worker, [tensor[start:end], device])
        th_pools.append(th)

    for i in range(nw):
        th_pools[i].start()
    for i in range(nw):
        th_pools[i].join()
    res = []
    for i in range(nw):
        sub_tensor = th_pools[i].get_result()
        res.append(sub_tensor)
    return torch.cat(res, dim=0) if len(res) > 1 else res[0]


def fast_norm(x, dim=0, eps=1.e-6):
    sum_x = x.sum(dim=dim, keepdim=True) + eps
    return x/sum_x


class Average(object):
    def __init__(self, tensor):
        self._sum = tensor
        self._count = 0

    def update(self, tensor):
        self._sum += tensor
        self._count += 1

    @property
    def value(self):
        return self._sum / max(1, self._count)