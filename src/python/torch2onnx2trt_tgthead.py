import ctypes
import onnx
import onnxsim
import os 
import torch 
import sys
import onnxruntime as ort 
import numpy as np

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from torch.onnx import register_custom_op_symbolic

import tensorrt as trt
from cuda import cudart
# from logger import set_logger



def export_norm_onnx(model, file, dummy_input:tuple, *args):
    
    # print(kwargs['opt_names'],kwargs['example_opts'])
    
    torch.onnx.export(
        model         = model, 
        args          = dummy_input,
        f             = file,
        input_names   = ["templates","dynamic_template","search-region","proposal","z_bbox","d_bbox"],
        output_names  = ["tgt_score"],
        opset_version = 12)

    print("Finished normal onnx export")

    model_onnx = onnx.load(file)

    # 检查导入的onnx model
    onnx.checker.check_model(model_onnx)

    # 使用onnx-simplifier来进行onnx的简化。
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    print(f'simple_finished!')
    onnx.save(model_onnx, file)

def trt_inference(engine, template, dynamic_template, search_region, proposal, z_bbox, d_bbox):
    nIO         = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput      = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context     = engine.create_execution_context()
    # context.set_input_shape(lTensorName[0], shape)

    # 初始化host端的数据，根据输入的shape大小来初始化值, 同时也把存储输出的空间存储下来
    bufferH     = []
    # bufferH.append(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    bufferH.append(template); bufferH.append(dynamic_template); bufferH.append(search_region);bufferH.append(proposal)
    bufferH.append(z_bbox); bufferH.append(d_bbox)
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))

    # 初始化device端的内存，根据host端的大小来分配空间
    bufferD     = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    # H2D, enqueue, D2H执行推理，并把结果返回
    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))
    context.execute_async_v3(0)
    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for b in bufferD:
        cudart.cudaFree(b)

    return nInput, nIO, bufferH

def onnx_run_check(file,model_ipt:tuple):
    # pytorch run
    # pytorch_model_opt = torch_model(model_ipt[0], model_ipt[1], model_ipt[2], model_ipt[3], model_ipt[4], model_ipt[5])
    # torch_opt_score = pytorch_model_opt.detach().cpu().numpy()
    # onnxruntime
    if file.endswith(".onnx"):
        sess = ort.InferenceSession(file)
        opt_score = sess.run(None, {'templates':model_ipt[0].cpu().numpy(),
                                    'dynamic_template':model_ipt[1].cpu().numpy(),
                                    "search-region":model_ipt[2].cpu().numpy(),
                                    "proposal":model_ipt[3].cpu().numpy(),
                                    "z_bbox":model_ipt[4].cpu().numpy(),
                                    "d_bbox":model_ipt[5].cpu().numpy()})
    elif file.endswith(".engine"):
        soFile = "/data1/dataset/wqj/project_2023/trt_experiments/chapter5/custom_prpool_plugin/lib/custom-plugin.so"
        ctypes.cdll.LoadLibrary(soFile)
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, '')
        ## 反序列化推理引擎
        with open(file, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print(f'Failed loading engine!')
        print(f'Succeeded loading engine!')
        nInput, nIO, bufferH = trt_inference(engine, model_ipt[0].cpu().numpy(), 
                                             model_ipt[1].cpu().numpy(), 
                                             model_ipt[2].cpu().numpy(),
                                             model_ipt[3].cpu().numpy(),
                                             model_ipt[4].cpu().numpy(),
                                             model_ipt[5].cpu().numpy())
        for i in range(nInput, nIO):
            opt_score_trt = bufferH[i]
            
        print(opt_score_trt)

if __name__ == "__main__":

    proposal = torch.rand(1,1,4).cuda(device=0)
    z = torch.rand(1,768,8,8).cuda(device=0)
    d = torch.rand(1,768,8,8).cuda(device=0)
    x = torch.rand(1,768,16,16).cuda(device=0)
    z_bbox = torch.rand(1,1,4).cuda(device=0)
    d_bbox = torch.rand(1,1,4).cuda(device=0)
    
    # save_file = "/data1/dataset/wqj/project_2023/anti-uav-exp/onnx_workspace/local_tracker_tgthead_sample.onnx"
    save_file = "/data1/dataset/wqj/project_2023/trt_experiments/chapter5/custom_prpool_plugin/models/engine/customPrpool.engine"

    onnx_run_check(save_file, model_ipt=(z, d, x, proposal, z_bbox, d_bbox))