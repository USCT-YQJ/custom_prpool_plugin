#include "custom-prpool-plugin.hpp"
#include <map>
#include <cstring>
#include <iostream>

/* CustomPrpool的核函数接口部分 */
// void CustomPrpoolImpl(const float* inputs, float* outputs, const float scalar, const float scale, const int nElements, cudaStream_t stream);
void customPrpoolImpl(const float* features, const float* rois, float* outputs, 
                      const int channels_, const int height_, const int width_, 
                      const int pooled_height_, const int pooled_width_,  const float spatial_scale_, const int top_count,cudaStream_t stream);

using namespace nvinfer1;

namespace
{
/******************************************************************/
/********************注册PluginCreator*****************************/
/******************************************************************/
REGISTER_TENSORRT_PLUGIN(CustomPrpoolPluginCreator);

/******************************************************************/
/*********************静态变量的申明*******************************/
/******************************************************************/
PluginFieldCollection   CustomPrpoolPluginCreator::mFC {};
std::vector<PluginField> CustomPrpoolPluginCreator::mAttrs;

/******************************************************************/
/*********************CustomPrpoolPlugin实现部分***********************/
/******************************************************************/

CustomPrpoolPlugin::CustomPrpoolPlugin(const std::string &name, int pooled_height, int pooled_width, float spatial_scale):
    mName(name)
{
    // std::cout << "CustomPrpoolPlugin_1" << std::endl;
    mParams.pooled_height = pooled_height;
    mParams.pooled_width = pooled_width;
    mParams.spatial_scale = spatial_scale;
}

CustomPrpoolPlugin::CustomPrpoolPlugin(const std::string &name, const void* buffer, size_t length):
    mName(name)
{
    // std::cout << "CustomPrpoolPlugin_2" << std::endl;
    memcpy(&mParams, buffer, sizeof(mParams));
}

CustomPrpoolPlugin::~CustomPrpoolPlugin()
{
    /* 这里的析构函数不需要做任何事情，生命周期结束的时候会自动调用terminate和destroy */
    // std::cout << "CustomPrpoolPlugin_3" << std::endl;
    return;
}

const char* CustomPrpoolPlugin::getPluginType() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    // std::cout << "getPluginType" << std::endl;
    return PLUGIN_NAME;
}

const char* CustomPrpoolPlugin::getPluginVersion() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    // std::cout << "getPluginVersion" << std::endl;
    return PLUGIN_VERSION;
}

int32_t CustomPrpoolPlugin::getNbOutputs() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    // std::cout << "getNbOutputs" << std::endl;
    return 1;
}

size_t CustomPrpoolPlugin::getSerializationSize() const noexcept
{
    /* 如果把所有的参数给放在mParams中的话, 一般来说所有插件的实现差不多一致 */
    // std::cout << "getSerializationSize" << std::endl;
    return sizeof(mParams);
}

const char* CustomPrpoolPlugin::getPluginNamespace() const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    // std::cout << "getPluginNamespace" << std::endl;
    return mNamespace.c_str();
}

DataType CustomPrpoolPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    // std::cout << "getOutputDataType" << std::endl;
    return inputTypes[0];
}

DimsExprs CustomPrpoolPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    // return inputs[0];
    // std::cout << "getOutputDimensions" << std::endl;
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    // const int roi = inputs[1].d[0];
    if (inputs[0].d[0]->isConstant()){
        output.d[0] = exprBuilder.constant(1);
        output.d[1] = exprBuilder.constant(768);
        output.d[2] = exprBuilder.constant(4);
        output.d[3] = exprBuilder.constant(4);
    }
    return output;
}

size_t CustomPrpoolPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    /* 一般来说会使用builder创建时用的workspaceSize所以这里一般什么都不做 */
    // std::cout << "getWorkspaceSize" << std::endl;
    return 0;
}

int32_t CustomPrpoolPlugin::initialize() noexcept
{
    /* 这个一般会根据情况而定，建议每个插件都有一个自己的实现 */
    // std::cout << "initialize" << std::endl;
    return 0;
}

void CustomPrpoolPlugin::terminate() noexcept 
{
    /* 
     * 这个是析构函数调用的函数。一般和initialize配对的使用
     * initialize分配多少内存，这里就释放多少内存
    */
//    std::cout << "terminate" << std::endl;
    return;
}

void CustomPrpoolPlugin::serialize(void *buffer) const noexcept
{
    /* 序列化也根据情况而定，每个插件自己定制 */
    // std::cout << "serialize" << std::endl;
    memcpy(buffer, &mParams, sizeof(mParams));
    return;

}

void CustomPrpoolPlugin::destroy() noexcept
{
    /* 一般来说所有插件的实现差不多一致 */
    // std::cout << "destroy" << std::endl;
    delete this;
    return;
}

int32_t CustomPrpoolPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, 
    const void* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) noexcept
{
    /*
     * Plugin的核心的地方。每个插件都有一个自己的定制方案
     * Plugin直接调用kernel的地方
    */
    // int nElements = 1;
    std::cout << "enqueue_tensorrt_cuda_infer" << std::endl;

    // for (int i = 0; i < inputDesc[0].dims.nbDims; i++){
    //     nElements *= inputDesc[0].dims.d[i];
    // }

    // CustomPrpoolImpl(
    //         static_cast<const float*>(inputs[0]),
    //         static_cast<float*>(outputs[0]), 
    //         mParams.scalar, 
    //         mParams.scale,
    //         nElements,
    //         stream);

    int nr_rois = inputDesc[1].dims.d[0];
    int nr_channels = inputDesc[0].dims.d[1];
    int height = inputDesc[0].dims.d[2];
    int width = inputDesc[0].dims.d[3];
    int top_count = nr_rois * nr_channels * mParams.pooled_height * mParams.pooled_width;
    // std::cout << inputDesc << std::endl;
    // std::cout << width << std::endl;
    customPrpoolImpl(
                static_cast<const float*>(inputs[0]),
                static_cast<const float*>(inputs[1]),
                static_cast<float*>(outputs[0]), 
                nr_channels,
                height,
                width,
                mParams.pooled_height, 
                mParams.pooled_width,
                mParams.spatial_scale,
                top_count,
                stream
                );

    return 0;
}

IPluginV2DynamicExt* CustomPrpoolPlugin::clone() const noexcept
{
    /* 克隆一个Plugin对象，所有的插件的实现都差不多*/
    // std::cout << "clone" << std::endl;
    auto p = new CustomPrpoolPlugin(mName, &mParams, sizeof(mParams));
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

bool CustomPrpoolPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    /* 
     * 设置这个Plugin支持的Datatype以及TensorFormat, 每个插件都有自己的定制
     * 作为案例展示，这个CustomPrpool插件只支持FP32，如果需要扩展到FP16以及INT8，需要在这里设置
    */
    // std::cout << "supportsFormatCombination" << std::endl;
    // switch (pos) {
    // case 0:
    //     return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
    // case 1:
    //     return inOut[1].type == DataType::kFLOAT && inOut[1].format == TensorFormat::kLINEAR;
    // default:
    //     return false;
    // }
    // return false;
    // 假设有两个输入一个输出
    // assert(0 <= pos && pos < 2);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    switch (pos) {
        case 0:
        return in[0].type == DataType::kFLOAT &&
                in[0].format == nvinfer1::TensorFormat::kLINEAR;
        case 1:
        return  in[1].type == DataType::kFLOAT &&
                in[1].format == nvinfer1::TensorFormat::kLINEAR;
        case 2:
        return out[0].type == in[0].type &&
                out[0].format == nvinfer1::TensorFormat::kLINEAR;
                }

}

void CustomPrpoolPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    /* 一般不需要做任何使用，所有插件实现都差不多 */
    // std::cout << "configurePlugin" << std::endl;
    return;
}
void CustomPrpoolPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    /* 所有插件的实现都差不多 */
    // std::cout << "setPluginNamespace" << std::endl;
    mNamespace = pluginNamespace;
    return;
}
void CustomPrpoolPlugin::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept 
{
    /* 一般不需要做任何使用，所有插件实现都差不多 */
    // std::cout << "attachToContext" << std::endl;
    return;
}
void CustomPrpoolPlugin::detachFromContext() noexcept 
{
    /* 一般不需要做任何使用，所有插件实现都差不多 */
    // std::cout << "detachFromContext" << std::endl;
    return;
}

/******************************************************************/
/*********************CustomPrpoolPluginCreator部分********************/
/******************************************************************/

CustomPrpoolPluginCreator::CustomPrpoolPluginCreator()
{
    /* 
     * 每个插件的Creator构造函数需要定制，主要就是获取参数以及传递参数
     * 初始化creator中的PluginField以及PluginFieldCollection
     * - PluginField::            负责获取onnx中的参数
     * - PluginFieldCollection：  负责将onnx中的参数传递给Plugin
    */
    // std::cout << "CustomPrpoolPluginCreator" << std::endl;
    mAttrs.emplace_back(PluginField("pooled_height", nullptr, PluginFieldType::kINT32, 1));
    mAttrs.emplace_back(PluginField("pooled_width", nullptr, PluginFieldType::kINT32, 1));
    mAttrs.emplace_back(PluginField("spatial_scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mAttrs.size();
    mFC.fields   = mAttrs.data();
}

CustomPrpoolPluginCreator::~CustomPrpoolPluginCreator()
{
    /* 一般不需要做任何使用，所有插件实现都差不多 */
}

const char* CustomPrpoolPluginCreator::getPluginName() const noexcept
{
    /* 所有插件实现都差不多 */
    // std::cout << "getPluginName" << std::endl;
    return PLUGIN_NAME;
}

const char* CustomPrpoolPluginCreator::getPluginVersion() const noexcept 
{
    /* 所有插件实现都差不多 */
    // std::cout << "getPluginVersion" << std::endl;
    return PLUGIN_VERSION;
}

const char* CustomPrpoolPluginCreator::getPluginNamespace() const noexcept
{
    /* 所有插件实现都差不多 */
    // std::cout << "getPluginNamespace" << std::endl;
    return mNamespace.c_str();
}

IPluginV2* CustomPrpoolPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept 
{
    /*
     * 通过Creator创建一个Plugin的实现，这个时候会通过mFC中取出需要的参数, 并实例化一个Plugin
     * 这个案例中，参数有scalar和scale两个参数。从fc中取出来对应的数据来初始化这个plugin
    */
    // std::cout << "createPlugin" << std::endl;
    int pooled_height = 4;
    int pooled_width  = 4;
    float spatial_scale = 1.0;
    // std::map<std::string, float*> paramMap = {{"scalar", &scalar}, {"scale", &scale}};

    // for (int i = 0; i < fc->nbFields; i++) {
    //     if (paramMap.find(fc->fields[i].name) != paramMap.end()){
    //         *paramMap[fc->fields[i].name] = *reinterpret_cast<const float*>(fc->fields[i].data);
    //     }
    // }
    // std::cout << pooled_height << std::endl;
    // std::cout << name << std::endl;
    return new CustomPrpoolPlugin(name, pooled_height, pooled_width,spatial_scale);
}

IPluginV2* CustomPrpoolPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    /* 反序列化插件其实就是实例化一个插件，所有插件实现都差不多 */
    // std::cout << "deserializePlugin" << std::endl;
    return new CustomPrpoolPlugin(name, serialData, serialLength);
}

void CustomPrpoolPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    /* 所有插件实现都差不多 */
    // std::cout << "setPluginNamespace" << std::endl;
    mNamespace = pluginNamespace;
    return;
}

const PluginFieldCollection* CustomPrpoolPluginCreator::getFieldNames() noexcept
{
    /* 所有插件实现都差不多 */
    // std::cout << "getFieldNames" << std::endl;
    return &mFC;
}

} // namespace custom
