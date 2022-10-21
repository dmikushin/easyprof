#ifndef EASYPROF_H
#define EASYPROF_H

#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
using gpuError_t = hipError_t;
using gpuStream_t = hipStream_t;
using gpuModule_t = hipModule_t;
using gpuFunction_t = hipFunction_t;
using gpuEvent_t = hipEvent_t;
static const auto gpuSuccess = hipSuccess;
#define __gpuRegisterFunction __hipRegisterFunction
#define gpuModuleGetFunction hipModuleGetFunction
#define GPU_FUNC_ATTRIBUTE_NUM_REGS HIP_FUNC_ATTRIBUTE_NUM_REGS
#define gpuFuncGetAttribute(...) hipFuncGetAttribute(__VA_ARGS__)
#define gpuGetLastError() hipGetLastError()
#define gpuGetErrorString(...) hipGetErrorString(__VA_ARGS__)
#define LIBGPURT "/opt/rocm/hip/lib/libamdhip64.so"
#else
#include <cuda.h>
#include <cuda_runtime.h>
using gpuError_t = CUresult;
using gpuStream_t = CUstream;
using gpuModule_t = CUmodule;
using gpuFunction_t = CUfunction;
static const auto gpuSuccess = CUDA_SUCCESS;
#define gpuModuleGetFunction cuModuleGetFunction
#define GPU_FUNC_ATTRIBUTE_NUM_REGS CU_FUNC_ATTRIBUTE_NUM_REGS
#define gpuFuncGetAttribute(...) cuFuncGetAttribute(__VA_ARGS__)
#define gpuGetLastError() cudaGetLastError()
inline const char* gpuGetErrorString(CUresult err)
{
	const char* errStr;
	cuGetErrorString(err, &errStr);
	return errStr;
}
#define gpuLaunchKernel(...) cuLaunchKernel(__VA_ARGS__)
#define LIBGPU "/usr/lib/x86_64-linux-gnu/libcuda.so"
#define LIBGPURT "/usr/local/cuda/lib64/libcudart.so"
#endif

#define LIBDL "libdl.so"

#define LOG(...) do { printf(__VA_ARGS__); printf("\n"); } while (0)

extern void* libdl;
extern void* libgpu;
extern void* libgpurt;

#ifdef __HIPCC__
#define DriverLibraryPrefix hip
#define RuntimeLibraryPrefix hip
#define SystemLibraryPrefix
#else
#define DriverLibraryPrefix cu
#define RuntimeLibraryPrefix cuda
#define SystemLibraryPrefix
#endif

#define str_prefix(prefix) #prefix
#define str_api_name(name, prefix) #prefix #name
#define api_name(name, prefix) prefix##name

extern std::map<std::string, void*> dlls;

const char* __dll(const char* prefix);

typedef void* (*dlsym_t)(void*, const char*);

// Intercept dynamically-loaded API calls, such as in the case
// of statically-linked cudart.
class SymbolLoader
{
	dlsym_t dlsym_real;

public :

	static void* get(void* lib, const char* sym);

	SymbolLoader();
};

struct GPUapiFunc
{
	const char* name;
	void* ptr;
};

#define GPU_API_NAME(name) { #name, reinterpret_cast<void*>(name) }

extern std::vector<GPUapiFunc> gpuAPI;

struct GPUfunction
{
	const void* deviceFun;
	std::string deviceName;
	int nregs;
};

using Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct Launch
{
	const void* deviceFun;
	Timestamp begin, end;
	dim3 numBlocks, dimBlocks;
	unsigned int sharedMemBytes;
};

// Maintaining the proper order of destruction.
class Profiler
{
	Profiler();

	// We don't use keys, as the kernels and kernels launches
	// are already keyed by the device function pointer.
	std::unordered_map<const void*, GPUfunction> funcs;
	
	// This is the current tape for kernel launches. Make it a flat
	// array to maximize the speed. Furthermore, we make sure the
	// vector is not reallocated, and the data record pointer passed
	// to the stream callbacks remains valid.
	std::vector<Launch>* launches;
	
	// Our memory is infinite, we just grow it once in a while.
	// Later we may dynamically offload old recordings somewhere,
	// perhaps send them over via a socket.
	std::list<std::vector<Launch>> archive;
	
	void rotateArchive();

public :

	~Profiler();
	
	Profiler(const Profiler&) = delete;

	Profiler& operator=(const Profiler&) = delete;

	static Profiler& get();

	template<typename F>
	void addKernel(const void* f, F&& func)
	{
		if (funcs.find(f) != funcs.end()) return;
		
		funcs.emplace(f, func());
	}

	Launch* start(gpuStream_t stream, const void* deviceFun,
		const dim3& gridDim, const dim3& blockDim, unsigned int sharedMemBytes);

	void stop(gpuStream_t stream, Launch* launch);
};

#endif // EASYPROF_H

