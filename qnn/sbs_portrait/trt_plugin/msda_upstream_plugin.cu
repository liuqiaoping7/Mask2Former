// TensorRT plugin "MSDeformAttnUpstream_TRT" (version 1): wraps the project's own MSDeformAttn CUDA
// forward kernel (mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_im2col_cuda.cuh, included
// unchanged) as an IPluginV3, for comparison with TensorRT's built-in MultiscaleDeformableAttnPlugin_TRT.
//
// Inputs (same layout as the upstream MSDeformAttnFunction / the built-in plugin):
//   0 value              (N, Len_in, M, D)        float32
//   1 spatial_shapes     (L, 2)                   int64   (upstream dtype: no cast needed)
//   2 level_start_index  (L,)                     int64
//   3 sampling_locations (N, Len_q, M, L, P, 2)   float32
//   4 attention_weights  (N, Len_q, M, L, P)      float32
// Output: (N, Len_q, M * D) float32 -- what the upstream op returns (the built-in plugin returns
// (N, Len_q, M, D), reshaped afterwards in the graph).
//
// The upstream kernel only dispatches float / double, so the plugin is FP32 only: in an FP16 engine
// TensorRT inserts casts around it (the MSDA core then runs in FP32, like the PyTorch FP16 path).
//
// IPluginV3 = one object exposing three capabilities:
//   core    (name / version / namespace)            -> IPluginV3OneCore
//   build   (output types / shapes, formats, ws)     -> IPluginV3OneBuild
//   runtime (shape change, enqueue, serialization)   -> IPluginV3OneRuntime
// The creator (IPluginCreatorV3One) is registered with the global plugin registry from a static
// initializer, so loading the library with dlopen(RTLD_GLOBAL) before parsing / deserializing is enough;
// the ONNX parser maps a node whose op_type equals the plugin name to it.
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "ms_deform_im2col_cuda.cuh"

using namespace nvinfer1;

namespace
{
char const* const kNAME = "MSDeformAttnUpstream_TRT";
char const* const kVERSION = "1";
char const* const kNAMESPACE = "";

class MSDAUpstreamPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    MSDAUpstreamPlugin()
    {
        mFields.nbFields = 0;  // no attributes: everything follows from the input shapes
        mFields.fields = nullptr;
    }

    // ---------------------------------------------------------------- IPluginV3
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override { return new MSDAUpstreamPlugin(*this); }

    // ---------------------------------------------------------------- core
    AsciiChar const* getPluginName() const noexcept override { return kNAME; }
    AsciiChar const* getPluginVersion() const noexcept override { return kVERSION; }
    AsciiChar const* getPluginNamespace() const noexcept override { return kNAMESPACE; }

    // ---------------------------------------------------------------- build
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t, DataType const*, int32_t) const noexcept override
    {
        outputTypes[0] = DataType::kFLOAT;
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* outputs,
        int32_t, IExprBuilder& exprBuilder) noexcept override
    {
        if (nbInputs != 5 || inputs[0].nbDims != 4 || inputs[3].nbDims != 6)
        {
            return -1;
        }
        outputs[0].nbDims = 3;
        outputs[0].d[0] = inputs[0].d[0];                                                   // N
        outputs[0].d[1] = inputs[3].d[1];                                                   // Len_q
        outputs[0].d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);  // M * D
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t, int32_t) noexcept override
    {
        PluginTensorDesc const& d = inOut[pos].desc;
        if (d.format != TensorFormat::kLINEAR)
        {
            return false;
        }
        if (pos == 1 || pos == 2)
        {
            return d.type == DataType::kINT64;
        }
        return d.type == DataType::kFLOAT;  // value, locations, weights and the output
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override
    {
        return 0;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
    }

    // ---------------------------------------------------------------- runtime
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        Dims const& v = inputDesc[0].dims;    // N, Len_in, M, D
        Dims const& loc = inputDesc[3].dims;  // N, Len_q, M, L, P, 2
        int const N = static_cast<int>(v.d[0]), lenIn = static_cast<int>(v.d[1]), M = static_cast<int>(v.d[2]),
                  D = static_cast<int>(v.d[3]);
        int const lenQ = static_cast<int>(loc.d[1]), L = static_cast<int>(loc.d[3]), P = static_cast<int>(loc.d[4]);
        // the upstream host function loops over the batch in im2col_step chunks; the kernel itself takes the
        // batch size, so one launch over the whole batch is the same computation
        ms_deformable_im2col_cuda<float>(stream, static_cast<float const*>(inputs[0]),
            static_cast<int64_t const*>(inputs[1]), static_cast<int64_t const*>(inputs[2]),
            static_cast<float const*>(inputs[3]), static_cast<float const*>(inputs[4]), N, lenIn, M, D, L, lenQ, P,
            static_cast<float*>(outputs[0]));
        return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFields; }

private:
    PluginFieldCollection mFields{};
};

class MSDAUpstreamCreator : public IPluginCreatorV3One
{
public:
    MSDAUpstreamCreator()
    {
        mFields.nbFields = 0;
        mFields.fields = nullptr;
    }

    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const*, TensorRTPhase) noexcept override
    {
        return new MSDAUpstreamPlugin();
    }

    PluginFieldCollection const* getFieldNames() noexcept override { return &mFields; }
    AsciiChar const* getPluginName() const noexcept override { return kNAME; }
    AsciiChar const* getPluginVersion() const noexcept override { return kVERSION; }
    AsciiChar const* getPluginNamespace() const noexcept override { return kNAMESPACE; }

private:
    PluginFieldCollection mFields{};
};

struct Registrar
{
    Registrar()
    {
        static MSDAUpstreamCreator creator;
        if (!getPluginRegistry()->registerCreator(creator, kNAMESPACE))
        {
            std::fprintf(stderr, "[msda_upstream_plugin] registerCreator failed\n");
        }
    }
};
Registrar gRegistrar;
} // namespace
