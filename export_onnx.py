
import argparse
import os
from models import model_factory
from models.configs.detault import CONFIGS as config
from datasets.dataset_reader import DatasetReader
from preprocess.data_preprocess import TestTransform
import torch
import torch.nn as nn
from utils import check_point
from onnxsim import simplify


def _transform_weights_(m: nn.Module):
    if isinstance(m, torch.Tensor):
        print('type:%s, device:%s' % (m.dtype, m.device))
        m = m.cuda()

def _print_weights_(model):
    for k, v in model.named_parameters():
        print('name:%s, type:%s, device:%s' % (k, v.dtype, v.device))


def setup(args):
    cfg = config.clone()
    if len(args.model_config) > 0:
        cfg.merge_from_file(args.model_config)
    device = torch.device(cfg.DEVICE) if torch.cuda.is_available() else torch.device('cpu')
    cfg.update({'DEVICE': device})
    model = model_factory.create_model(cfg)
    dataset = DatasetReader(cfg.DATASET.PATH, cfg,
                            augment=TestTransform(cfg.INPUT_SIZE[0]), is_training=False, split='test')
    model.to(device)
    model.eval()
    return model, dataset, cfg


def export_onnx(model, dataset, cfg):

    # save_dir = os.path.join(cfg.TRAINING.WEIGHTS, cfg.MODEL.BACKBONE)
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    checkpointer = check_point.CheckPointer(model,
                                            # save_dir=save_dir,
                                            mode='state-dict',
                                            device=cfg.DEVICE)

    ckpt = checkpointer.load(cfg.DETECTOR.CHECKPOINT, use_latest=False, load_solver=False)
    del ckpt
    w, h = dataset._img_size
    # img = torch.rand(1, 3, h, w).to(cfg.DEVICE)
    img, targets, _, _, _ = dataset[0]
    img = img.unsqueeze(dim=0).to(cfg.DEVICE)
    half = cfg.DEVICE.type != 'cpu'
    half = False
    if half:
        model.half()  # to FP16
        img = img.half()
    else:
        model.to(torch.float32)
    # _print_weights_(model)
    model.export = True
    _ = model(img)  # dry run
    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = cfg.DETECTOR.CHECKPOINT.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx
        print(onnx.__version__)
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = cfg.DETECTOR.CHECKPOINT.replace('.pt', '.onnx')  # filename
        # model.fuse()  # only for ONNX
        # model.to(cfg.DEVICE)
        # model.to(torch.float32)
        # _print_weights_(model)
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['outputs'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx_model, check = simplify(onnx_model)
        onnx.checker.check_model(onnx_model)  # check onnx model
        onnx.save(onnx_model, f)
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RTM3D Detecting")
    parser.add_argument("--model-config", default="", help="specific model config path")
    args = parser.parse_args()
    model, dataset, cfg = setup(args)
    export_onnx(model, dataset, cfg)


