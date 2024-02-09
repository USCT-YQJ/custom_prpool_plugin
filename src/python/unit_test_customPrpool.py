import ctypes
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import numpy as np
import tensorrt as trt
import logging

from trt_model import test_logger, console_handler, file_handler
from trt_model import build_network, inference, validation


ONNX_PATH = "/data1/dataset/wqj/project_2023/trt_experiments/chapter5/custom_prpool_plugin/models/onnx/local_tracker_tgthead_sample.onnx"

def CustomScalarCPU(inputH, scalar, scale):
    return [(inputH[0] + scalar) * scale]

def getCustomScalarPlugin(pooled_height, pooled_width, spatial_scale) -> trt.tensorrt.IPluginV2:
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "customPrpool":
            parameterList = []
            parameterList.append(trt.PluginField("pooled_height", np.int32(pooled_height), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("pooled_width", np.int32(pooled_width), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("spatial_scale", np.float32(spatial_scale), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None


def customScalarTest(pooled_height, pooled_width, spatial_scale):
    current_path = os.path.dirname(__file__)
    soFile       = "/data1/dataset/wqj/project_2023/trt_experiments/chapter5/custom_prpool_plugin/lib/custom-plugin.so"
    trtFile      = current_path + "/../../models/engine/customPrpool.engine"
    # testCase     = "<shape=%s,scalar=%f,scale=%f>" % (shape, scalar, scale)

    ctypes.cdll.LoadLibrary(soFile)
    plugin = getCustomScalarPlugin(pooled_height=pooled_height, pooled_width=pooled_width, spatial_scale=spatial_scale)
    # test_logger.info("Test '%s':%s" % (plugin.plugin_type, testCase))

    #################################################################
    ################### 从这里开始是builder的部分 ######################
    #################################################################
    engine = build_network(trtFile, plugin, onnx_file_path=ONNX_PATH)
    if (engine == None):
        exit()

    #################################################################
    ################### 从这里开始是infer的部分 ########################
    #################################################################
    # nInput, nIO, bufferH = inference(engine)

    #################################################################
    ################# 从这里开始是validation的部分 #####################
    #################################################################
    # outputCPU = CustomScalarCPU(bufferH[:nInput], scalar, scale)
    # res = validation(nInput, nIO, bufferH, outputCPU)

    # if (res):
    #     test_logger.info("Test '%s':%s finish!\n" % (plugin.plugin_type, testCase))
    # else:
    #     test_logger.error("Test '%s':%s failed!\n" % (plugin.plugin_type, testCase))
    #     exit()

def unit_test():
    customScalarTest(4, 4, 1.0)

if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    np.random.seed(1)

    test_logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    test_logger.info("Starting unit test...")
    unit_test()
    test_logger.info("All tests are passed!!")
