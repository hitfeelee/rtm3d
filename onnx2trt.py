import argparse
import tensorrt as trt
import os
from utils import trt_utils


def ONNX2TRT(args, calib=None):
    ''' convert onnx to tensorrt engine, use mode of ['fp32', 'fp16', 'int8']
    :return: trt engine
    '''

    assert args.mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"

    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(G_LOGGER) as builder, builder.create_network(trt_utils.EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:

        builder.max_batch_size = args.batch_size
        builder.max_workspace_size = 1 << 30
        if args.mode.lower() == 'int8':
            assert (builder.platform_has_fast_int8 == True), "not support int8"
            builder.int8_mode = True
            builder.int8_calibrator = calib
        elif args.mode.lower() == 'fp16':
            assert (builder.platform_has_fast_fp16 == True), "not support fp16"
            builder.fp16_mode = True

        print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
        with open(args.onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print('Completed parsing of ONNX file')

        print('Building an engine from file {}; this may take a while...'.format(args.onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Created engine success! ")

        # 保存计划文件
        print('Saving TRT engine file to path {}...'.format(args.engine_file_path))
        with open(args.engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print('Engine file has already saved to {}!'.format(args.engine_file_path))
        return engine


TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # network.get_input(0).shape = [1, 3, 608, 608]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RTM3D Detecting")
    parser.add_argument("--onnx_file_path", default="", help="specific onnx file path")
    parser.add_argument("--engine_file_path", default="./", help="specific output engine path")
    parser.add_argument("--mode", default="fp16", help="specific model data type")
    parser.add_argument("--batch_size", default=1, help="specific model batch size")
    args = parser.parse_args()
    ONNX2TRT(args)
    # get_engine(args.onnx_file_path, args.engine_file_path)