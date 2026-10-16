#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <cstring>
#include <new>

using namespace nvinfer1;
static_assert(NV_TENSORRT_MAJOR == 11, "The VOSR plugin requires TensorRT 11.");
extern "C" int launchPack(void const* const*, void* const*, int, cudaStream_t);

namespace {
bool vector64(Dims const& d) { return d.nbDims == 1 && d.d[0] == 64; }
bool table(Dims const& d, int64_t n) {
    return d.nbDims == 2 && d.d[0] == n && d.d[1] == 64;
}
bool shape(PluginTensorDesc const* in, PluginTensorDesc const* out) {
    auto const& x = in[0].dims;
    if (x.nbDims != 3 || x.d[0] != 1 || x.d[2] != 4608 ||
        x.d[1] < 1 || x.d[1] > 1152 || !vector64(in[1].dims) ||
        !vector64(in[2].dims) || !table(in[3].dims, x.d[1]) ||
        !table(in[4].dims, x.d[1])) return false;
    for (int i = 0; i < 5; ++i)
        if (in[i].type != DataType::kBF16 || in[i].format != TensorFormat::kLINEAR)
            return false;
    for (int i = 0; i < 3; ++i) {
        auto const& d = out[i].dims;
        if (out[i].type != DataType::kBF16 || out[i].format != TensorFormat::kLINEAR ||
            d.nbDims != 4 || d.d[0] != 1 || d.d[1] != 24 ||
            d.d[2] != x.d[1] || d.d[3] != 64) return false;
    }
    return true;
}

class QkvPack final : public IPluginV3, public IPluginV3OneCore,
                      public IPluginV3OneBuild, public IPluginV3OneRuntime {
    int32_t mode = 0;
    PluginField field{"mode", &mode, PluginFieldType::kINT32, 1};
    PluginFieldCollection fields{1, &field};
public:
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
        if (t == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new (std::nothrow) QkvPack(); }
    char const* getPluginName() const noexcept override { return "VosrQkvPack"; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return 3; }
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t n,
        DynamicPluginTensorDesc const*, int32_t m) noexcept override {
        if (n != 5 || m != 3 || in[0].desc.dims.nbDims != 3 ||
            !vector64(in[1].desc.dims) || !vector64(in[2].desc.dims) ||
            in[3].desc.dims.nbDims != 2 || in[4].desc.dims.nbDims != 2) return -1;
        return 0;
    }
    bool supportsFormatCombination(int32_t p, DynamicPluginTensorDesc const* io,
        int32_t n, int32_t m) noexcept override {
        return n == 5 && m == 3 && p >= 0 && p < 8 &&
            io[p].desc.type == DataType::kBF16 && io[p].desc.format == TensorFormat::kLINEAR;
    }
    int32_t getOutputDataTypes(DataType* types, int32_t m,
        DataType const*, int32_t n) const noexcept override {
        if (n != 5 || m != 3) return -1;
        for (int i = 0; i < 3; ++i) types[i] = DataType::kBF16;
        return 0;
    }
    int32_t getOutputShapes(DimsExprs const* in, int32_t n, DimsExprs const*, int32_t,
        DimsExprs* out, int32_t m, IExprBuilder& e) noexcept override {
        if (n != 5 || m != 3 || in[0].nbDims != 3) return -1;
        for (int i = 0; i < 3; ++i) {
            out[i].nbDims = 4;
            out[i].d[0] = in[0].d[0]; out[i].d[1] = e.constant(24);
            out[i].d[2] = in[0].d[1]; out[i].d[3] = e.constant(64);
        }
        return 0;
    }
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t n,
        PluginTensorDesc const* out, int32_t m) noexcept override {
        return n == 5 && m == 3 && shape(in, out) ? 0 : -1;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const* out,
        void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        if (!shape(in, out)) return -1;
        return launchPack(inputs, outputs, static_cast<int>(in[0].dims.d[1]), stream);
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &fields; }
};

class QkvPackCreator final : public IPluginCreatorV3One {
    PluginField field{"mode", nullptr, PluginFieldType::kINT32, 1};
    PluginFieldCollection fields{1, &field};
public:
    char const* getPluginName() const noexcept override { return "VosrQkvPack"; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fields; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        if (!fc || fc->nbFields != 1 || !fc->fields) return nullptr;
        auto const& f = fc->fields[0];
        if (!f.name || std::strcmp(f.name, "mode") || f.type != PluginFieldType::kINT32 ||
            f.length != 1 || !f.data || *static_cast<int32_t const*>(f.data) != 0) return nullptr;
        return new (std::nothrow) QkvPack();
    }
};
REGISTER_TENSORRT_PLUGIN(QkvPackCreator);
}
