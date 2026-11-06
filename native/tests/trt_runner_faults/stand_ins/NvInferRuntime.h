// Stand-in for what runner.cpp calls of TensorRT, on the CPU (faults.cpp).
#pragma once
#include "cuda_runtime_api.h"
#include <cstddef>
#include <cstdint>
#define NV_TENSORRT_MAJOR 11
namespace nvinfer1 {
using AsciiChar = char;
class ILogger {
public:
    enum class Severity { kINTERNAL_ERROR, kERROR, kWARNING, kINFO, kVERBOSE };
    virtual void log(Severity, AsciiChar const*) noexcept = 0;
    virtual ~ILogger() = default;
};
enum class DataType { kFLOAT, kHALF, kBF16, kINT8 };
enum class TensorIOMode { kNONE, kINPUT, kOUTPUT };
enum class OptProfileSelector { kMIN, kOPT, kMAX };
enum class ExecutionContextAllocationStrategy { kSTATIC, kUSER_MANAGED };
struct Dims {
    int32_t nbDims = 0;
    int64_t d[8] = {};
};
struct Dims4 : Dims {
    Dims4(int64_t a, int64_t b, int64_t c, int64_t e) {
        nbDims = 4;
        d[0] = a;
        d[1] = b;
        d[2] = c;
        d[3] = e;
    }
};
class IExecutionContext {
public:
    virtual ~IExecutionContext() = default;
    virtual bool setInputShape(char const*, Dims const&) = 0;
    virtual Dims getTensorShape(char const*) const = 0;
    virtual std::size_t updateDeviceMemorySizeForShapes() = 0;
    virtual void setDeviceMemoryV2(void*, int64_t) = 0;
    virtual bool setTensorAddress(char const*, void*) = 0;
    virtual bool enqueueV3(cudaStream_t) = 0;
};
class ICudaEngine {
public:
    virtual ~ICudaEngine() = default;
    virtual int getNbIOTensors() const = 0;
    virtual char const* getIOTensorName(int) const = 0;
    virtual TensorIOMode getTensorIOMode(char const*) const = 0;
    virtual Dims getTensorShape(char const*) const = 0;
    virtual DataType getTensorDataType(char const*) const = 0;
    virtual Dims getProfileShape(char const*, int, OptProfileSelector) const = 0;
    virtual IExecutionContext* createExecutionContext(ExecutionContextAllocationStrategy) = 0;
};
class IRuntime {
public:
    virtual ~IRuntime() = default;
    virtual ICudaEngine* deserializeCudaEngine(void const*, std::size_t) = 0;
};
IRuntime* createInferRuntime(ILogger&);
}  // namespace nvinfer1
bool initLibNvInferPlugins(void*, char const*);
int32_t getInferLibVersion();
