#include "easyprof.h"

#include <cxxabi.h>
#include <dlfcn.h>
#include <functional>

#define GPU_FUNC_LAUNCH_BEGIN(prefix, __stream, __function, \
	RetTy, name, ...) \
	extern "C" \
	RetTy api_name(name, prefix)(__VA_ARGS__) \
	{ \
		gpuStream_t __s = static_cast<gpuStream_t>(__stream); \
		return gpuFuncLaunch<RetTy>( \
			__dll(str_prefix(prefix)), str_api_name(name, prefix), __s, __function,

#define GPU_FUNC_LAUNCH_END(...) \
			__VA_ARGS__); \
	}

// CUDA/HIP APIs can execute a user callback function, when the corresponding
// stream reaches the point of interest. We use this feature to track kernels
// execution in a simple way.
#ifdef __HIPCC__
static void profilerTimerSync(hipStream_t stream, hipError_t status, void *userData)
#else
static void profilerTimerSync(CUstream stream, CUresult status, void *userData)
#endif
{
	if (Profiler::get().timer->isTiming())
		Profiler::get().timer->sync(stream);
}

// This is a reverse-engineering of some internal CUDA structures,
// in order to reach out some data, most importantly to the kernel name.

struct kernel
{
	uint32_t v0;
	uint32_t v1;
	uint32_t v2;
	uint64_t v3;
	uint32_t v4;
	uint32_t v5;
	uint32_t v6;
	uint32_t v7;
	uint32_t v8;
	void *module;
	uint32_t size;
	uint32_t v9;
	void *p1;   
};

struct dummy1
{
	void *p0;
	void *p1;
	uint64_t v0;
	uint64_t v1;
	void *p2;
};

struct CUfunc_st
{
	uint32_t v0;
	uint32_t v1;
	char *name;
	uint32_t v2;
	uint32_t v3;
	uint32_t v4;
	uint32_t v5;
	struct kernel *kernel;
	void *p1;
	void *p2;
	uint32_t v6;
	uint32_t v7;
	uint32_t v8;
	uint32_t v9;
	uint32_t v10;
	uint32_t v11;
	uint32_t v12;
	uint32_t v13;
	uint32_t v14;
	uint32_t v15;
	uint32_t v16;
	uint32_t v17;
	uint32_t v18;
	uint32_t v19;
	uint32_t v20;
	uint32_t v21;
	uint32_t v22;
	uint32_t v23;
	struct dummy1 *p3;
};

template<typename RetTy, typename Function, typename... Args>
RetTy gpuFuncLaunch(const std::string dll, const std::string sym, gpuStream_t stream, Function f,
	unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, Args... args)
{
	void* handle = nullptr;
	{
		auto it = dlls.find(dll);
		if (it != dlls.end())
			handle = it->second;
		else
		{
			handle = dlopen(dll.c_str(), RTLD_NOW | RTLD_GLOBAL);
			if (!handle)
			{
				LOG("Error loading %s: %s", dll.c_str(), dlerror());
				abort();
			}
			dlls.insert(std::make_pair(dll, handle));
		}
	}

	using Func = RetTy (*)(Args...);
	
	static Func funcReal = nullptr;
	if (!funcReal)
	{
		funcReal = (Func)SymbolLoader::get(handle, sym.c_str());
		if (!funcReal)
		{
			LOG("Error loading %s : %s", sym.c_str(), dlerror());
			abort();
		}
	}

	auto it = Profiler::get().funcs.find(reinterpret_cast<const void*>(f));
	if (it == Profiler::get().funcs.end())
	{
		struct CUfunc_st *pFunc = (struct CUfunc_st *)f;
		struct kernel *pKernel = pFunc->kernel;
#ifdef __CUDACC__
		int status;    
		char* name = abi::__cxa_demangle(pFunc->name, 0, 0, &status);	
		auto deviceName = status ? pFunc->name : name;
#else
		const char* deviceName = "(unknown)";
#endif
		// Get the kernel register count.
		int nregs = 0;
#ifdef __CUDACC__
		if (cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, pFunc) != CUDA_SUCCESS)
		{
			fprintf(stderr, "Could not read the number of registers for function \"%s\"\n", deviceName);
			auto err = gpuGetLastError();
		}
#else
		struct gpuFuncAttributes attrs;
		if (gpuFuncGetAttributes(&attrs, (void*)f) != gpuSuccess)
		{
			fprintf(stderr, "Could not read the number of registers for function \"%s\"\n", deviceName);
			auto err = gpuGetLastError();
		}
		else
		{
			nregs = attrs.numRegs;
		}
#endif
		auto result = Profiler::get().funcs.emplace(reinterpret_cast<const void*>(f),
			std::make_shared<GPUfunction>(GPUfunction
		{
			/* std::string deviceName; */      deviceName,
			/* char* deviceFun; */             f,
#ifdef __CUDACC__
			/* void* module */                 pKernel->module,
#else
			/* void* module */                 nullptr,
#endif
			/* unsigned int sharedMemBytes; */ sharedMemBytes,
			/* int nregs; */                   nregs
		}));

		it = result.first;
	}

	// Call the real function.
	auto result = std::invoke(funcReal, args...);

	auto& func = it->second;
	auto& name = func->deviceName;

	// Start profiling the newly-launched kernel.
	if (Profiler::get().matcher->isMatching(name))
	{
		// Don't do anything else, if kernel launch was not successful.
		if (result != CUDA_SUCCESS) return result;
		
		if (Profiler::get().timer->isTiming())
		{
			Profiler::get().timer->measure(func.get(),
				dim3(gridDimX, gridDimY, gridDimZ),
				dim3(blockDimX, blockDimY, blockDimZ),
				stream);
#ifdef __CUDACC__			
			// Insert a callback into the same stream after the launch,
			// in order to have it to stop the time measurement.
			auto err = cuStreamAddCallback(stream, profilerTimerSync, /* userData = */ nullptr, 0);
#else
			// in order to have it to stop the time measurement.
			auto err = hipStreamAddCallback(stream, profilerTimerSync, /* userData = */ nullptr, 0);
#endif
		}
	}

	return result;
}

#if defined(__HIPCC__)

// HIP has multiple different API functions for kernel launching.

GPU_FUNC_LAUNCH_BEGIN(RuntimeLibraryPrefix, stream, f,
	gpuError_t, LaunchKernel,
	const void* f, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, gpuStream_t stream)
GPU_FUNC_LAUNCH_END(numBlocks.x, numBlocks.y, numBlocks.z,
	dimBlocks.x, dimBlocks.y, dimBlocks.z, sharedMemBytes,
	f, numBlocks, dimBlocks, args, sharedMemBytes, stream);

GPU_FUNC_LAUNCH_BEGIN(RuntimeLibraryPrefix, stream, f,
	gpuError_t, ExtLaunchKernel,
	const void* f, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, gpuStream_t stream,
	gpuEvent_t startEvent, gpuEvent_t stopEvent, int flags)
GPU_FUNC_LAUNCH_END(numBlocks.x, numBlocks.y, numBlocks.z,
	dimBlocks.x, dimBlocks.y, dimBlocks.z, sharedMemBytes,
	f, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags);

GPU_FUNC_LAUNCH_BEGIN(RuntimeLibraryPrefix, stream, f,
	gpuError_t, ModuleLaunchKernel,
	gpuFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
	gpuStream_t stream, void** kernelParams, void** extra)
GPU_FUNC_LAUNCH_END(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
	f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
	stream, kernelParams, extra);

#else

// CUDA has Runtime API and Driver API functions for kernel launching,
// but the former in turn calls the latter, so we need to handle the driver API only.

GPU_FUNC_LAUNCH_BEGIN(DriverLibraryPrefix, hStream, f,
	CUresult, LaunchKernel,
	CUfunction f,
	unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra)
GPU_FUNC_LAUNCH_END(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
	f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);

#endif

