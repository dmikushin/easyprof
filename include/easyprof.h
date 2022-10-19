#ifndef EASYPROF_H
#define EASYPROF_H

#include <chrono>
#include <list>
#include <memory>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
using gpuError_t = hipError_t;
using gpuStream_t = hipStream_t;
static const auto gpuSuccess = hipSuccess;
#define __gpuRegisterFunction __hipRegisterFunction
#define gpuFuncAttributes hipFuncAttributes
#define gpuFuncGetAttributes(...) hipFuncGetAttributes(__VA_ARGS__)
#define gpuGetLastError(...) hipGetLastError(__VA_ARGS__)
// In HIP, hipLaunchKernel talks to the AMD driver directly,
// not as CUDA, which redirects cudaLaunchKernel to cuLaunchKernel.
#define gpuLaunchKernel(...) hipLaunchKernel(__VA_ARGS__)
// HIP-enabled libraries may call two different flavors of hipLaunchKernel
// presented below.
#define gpuModuleLaunchKernel(...) hipModuleLaunchKernel(__VA_ARGS__)
#define gpuExtLaunchKernel(...) hipExtLaunchKernel(__VA_ARGS__)
// These two types are used by hipModuleLaunchKernel and hipExtLaunchKernel
#define gpuFunction_t hipFunction_t
#define gpuEvent_t hipEvent_t
#define gpuStreamSynchronize(...) hipStreamSynchronize(__VA_ARGS__)
// In HIP, the CUDA driver styled context management is deprecated.
// So unlike in CUDA where cudaDeviceSynchronize calls cuCtxSynchronize,
// here we hook hipStreamSynchronize directly. 
#define gpuDeviceSynchronize() hipDeviceSynchronize()
#define gpuMemcpyKind hipMemcpyKind
#define gpuMemcpy(...) hipMemcpy(__VA_ARGS__)
#define gpuMemcpyAsync(...) hipMemcpyAsync(__VA_ARGS__)
#define LIBGPURT "/opt/rocm/hip/lib/libamdhip64.so"
#else
#include <cuda.h>
#include <cuda_runtime.h>
using gpuError_t = cudaError_t;
using gpuStream_t = cudaStream_t;
static const auto gpuSuccess = cudaSuccess;
#define __gpuRegisterFunction __cudaRegisterFunction
#define gpuFuncAttributes cudaFuncAttributes
#define gpuFuncGetAttributes(...) cudaFuncGetAttributes(__VA_ARGS__)
#define gpuGetLastError(...) cudaGetLastError(__VA_ARGS__)
// In CUDA, cudaLaunchKernel calls cuLaunchKernel, so we hook just for the last one.
#define gpuLaunchKernel(...) cuLaunchKernel(__VA_ARGS__)
#define gpuStreamSynchronize(...) cudaStreamSynchronize(__VA_ARGS__)
// In CUDA, cudaDeviceSynchronize calls cuCtxSynchronize
#define gpuDeviceSynchronize() cuCtxSynchronize()
#define gpuMemcpyKind cudaMemcpyKind
#define gpuMemcpy(...) cudaMemcpy(__VA_ARGS__)
#define gpuMemcpyAsync(...) cudaMemcpyAsync(__VA_ARGS__)
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

#include <map>
#include <string>
#include <vector>

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
	std::string deviceName;
	void* deviceFun;
	void* module;
	unsigned int sharedMemBytes;
	int nregs;
};

class Matcher;
class Timer;

// Maintaining the proper order of destruction.
class Profiler
{
	Profiler();

public :

	~Profiler();
	
	Profiler(const Profiler&) = delete;

	Profiler& operator=(const Profiler&) = delete;

	static Profiler& get();

	std::map<void*, std::shared_ptr<GPUfunction>> funcs;

	Matcher* matcher;

	// TODO must support threads.
	Timer* timer;
};

class Matcher
{
	const std::map<void*, std::shared_ptr<GPUfunction>>& funcs;

	std::string pattern;

public :

	bool isMatching(const std::string& name);

	Matcher(const std::map<void*, std::shared_ptr<GPUfunction>>& funcs_);
};

class Timer
{
	const std::map<void*, std::shared_ptr<GPUfunction>>& funcs;

	bool timing = false;
	
	// Enforce stream synchronization, when measuring the kernel
	// execution time.
	bool synced = false;

	using launches =
		std::vector<
			std::tuple<
				std::chrono::time_point<std::chrono::high_resolution_clock>, // time begin
				std::chrono::time_point<std::chrono::high_resolution_clock>, // time end
				dim3, dim3, // gridDim, blockDim
				int // synchronization group index
			>
		>;

	std::map<
		gpuStream_t, // for each stream
		std::map<
			std::string, // for each kernel name
			std::tuple<
				unsigned int, // the number of kernels to sync
				const GPUfunction*, // the corresponding registered function
				launches, // launches for the current time interval
				std::shared_ptr<std::list<launches> > // archived launches from the previous live intervals
			>
		>
	> kernels;
	
	// Archived launches from the previous live intervals,
	// which is to be filled by a long-running app with many kernel launches.
	std::map<GPUfunction*, std::shared_ptr<std::list<launches> > > archive;
	
	int sync_group_index = 0;
	
public :

	bool isTiming();
	
	Timer(const std::map<void*, std::shared_ptr<GPUfunction>>& funcs_) ;

	void measure(const GPUfunction* func_,
		const dim3& gridDim, const dim3& blockDim, gpuStream_t stream);

	void sync(gpuStream_t stream);
	
	void sync();
	
	~Timer();
};

#endif // EASYPROF_H

