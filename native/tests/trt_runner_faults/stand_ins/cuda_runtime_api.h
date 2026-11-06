// Stand-in for what runner.cpp calls of CUDA, on the CPU, with faults on demand (faults.cpp).
#pragma once
#include <cstddef>
#include <cstdint>
enum cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorUnknown = 999 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
enum cudaStreamCaptureMode { cudaStreamCaptureModeThreadLocal };
constexpr unsigned cudaStreamNonBlocking = 1;
constexpr unsigned cudaEventDisableTiming = 2;
struct FakeStream;
struct FakeGraph;
struct FakeExec;
struct FakeEvent;
using cudaStream_t = FakeStream*;
using cudaGraph_t = FakeGraph*;
using cudaGraphExec_t = FakeExec*;
using cudaEvent_t = FakeEvent*;
char const* cudaGetErrorName(cudaError_t);
char const* cudaGetErrorString(cudaError_t);
cudaError_t cudaGetLastError();
cudaError_t cudaSetDevice(int);
cudaError_t cudaMalloc(void**, std::size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMallocHost(void**, std::size_t);
cudaError_t cudaFreeHost(void*);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaMemcpyAsync(void*, void const*, std::size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMemsetAsync(void*, int, std::size_t, cudaStream_t);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned);
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
