// TODO 2) Different socket PORT numbers for HIP (2485) and CUDA (2788)
// TODO 3) FREEZE_ON_START=1 env variable to wait for a start command in socket

#include "easyprof.h"

#include <dlfcn.h>
#include <functional>

#define GPU_FUNC_LAUNCH_BEGIN(prefix, __stream, __function, \
	linkage, RetTy, name, ...) \
	linkage \
	RetTy api_name(name, prefix)(__VA_ARGS__) \
	{ \
		gpuStream_t __s = static_cast<gpuStream_t>(__stream); \
		return gpuFuncLaunch<RetTy>( \
			__dll(str_prefix(prefix)), str_api_name(name, prefix), __s, __function,

#define GPU_FUNC_LAUNCH_END(...) \
			__VA_ARGS__); \
	}

// This is a reverse-engineering of some internal CUDA structures,
// in order to reach out to some data, most importantly the kernel name.

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

#ifdef __HIPCC__
extern const char* nameExtModuleLaunchKernel;
#endif

template<typename RetTy, typename Function, typename... Args>
RetTy gpuFuncLaunch(const std::string dll, std::string sym, gpuStream_t stream, Function f,
	unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, Args... args)
{
	static void* handle = nullptr;
	if (!handle)
	{
		auto it = dlls.find(dll);
		if (it != dlls.end())
			handle = it->second;
		else
		{
			handle = dlopen(dll.c_str(), RTLD_LAZY | RTLD_GLOBAL);
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
#ifdef __HIPCC__
		// Hack around hipExtModuleLaunchKernel, which is the one API function with C++ mangling.
		if (sym == "hipExtModuleLaunchKernel")
			sym = nameExtModuleLaunchKernel;
#endif
		funcReal = (Func)SymbolLoader::get(handle, sym.c_str());
		if (!funcReal)
		{
			LOG("Error loading %s : %s", sym.c_str(), dlerror());
			abort();
		}
	}

	// Start profiling the newly-launched kernel.
	// Insert a callback into the same stream before and after the launch,
	// in order to have the time measurement started and stopped.
	auto launch = Profiler::get().start(f,
		dim3(gridDimX, gridDimY, gridDimZ),
		dim3(blockDimX, blockDimY, blockDimZ),
		sharedMemBytes, stream);

	// Call the real kernel launch function.
	auto result = std::invoke(funcReal, args...);

	Profiler::get().stop(stream, launch);

	return result;
}

#if defined(__HIPCC__)

// HIP has multiple different API functions for kernel launching.
// In HIP, hipLaunchKernel talks to the AMD driver directly,
// not as CUDA, which redirects cudaLaunchKernel to cuLaunchKernel.
// HIP-enabled libraries may call all different flavors of launch
// functions presented below.
GPU_FUNC_LAUNCH_BEGIN(RuntimeLibraryPrefix, stream, f,
	extern "C", gpuError_t, LaunchKernel,
	const void* f, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, gpuStream_t stream)
GPU_FUNC_LAUNCH_END(numBlocks.x, numBlocks.y, numBlocks.z,
	dimBlocks.x, dimBlocks.y, dimBlocks.z, sharedMemBytes,
	f, numBlocks, dimBlocks, args, sharedMemBytes, stream);

GPU_FUNC_LAUNCH_BEGIN(RuntimeLibraryPrefix, stream, f,
	extern "C", gpuError_t, ExtLaunchKernel,
	const void* f, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, gpuStream_t stream,
	gpuEvent_t startEvent, gpuEvent_t stopEvent, int flags)
GPU_FUNC_LAUNCH_END(numBlocks.x, numBlocks.y, numBlocks.z,
	dimBlocks.x, dimBlocks.y, dimBlocks.z, sharedMemBytes,
	f, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags);

GPU_FUNC_LAUNCH_BEGIN(RuntimeLibraryPrefix, stream, f,
	extern "C", gpuError_t, ModuleLaunchKernel,
	gpuFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
	gpuStream_t stream, void** kernelParams, void** extra)
GPU_FUNC_LAUNCH_END(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
	f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
	stream, kernelParams, extra);

// This variant is used to deploy the assembly kernels generated in Tensile.
GPU_FUNC_LAUNCH_BEGIN(RuntimeLibraryPrefix, stream, f,
	HIP_PUBLIC_API, gpuError_t, ExtModuleLaunchKernel,
	gpuFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, size_t sharedMemBytes,
	gpuStream_t stream, void** kernelParams, void** extra,
	gpuEvent_t startEvent, gpuEvent_t stopEvent, uint32_t flags)
GPU_FUNC_LAUNCH_END(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
	f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
	stream, kernelParams, extra, startEvent, stopEvent, flags);
// Note unlike the others hipExtModuleLaunchKernel function (suprisingly) belongs to the C++ API,
// not C API. So we have to provide its mangled name to dlsym().
const char* nameExtModuleLaunchKernel =
	"_Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_tjjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j";

#else

// CUDA has Runtime API and Driver API functions for kernel launching,
// but the former in turn calls the latter, so we need to handle the driver API only.

GPU_FUNC_LAUNCH_BEGIN(DriverLibraryPrefix, hStream, f,
	extern "C", CUresult, LaunchKernel,
	CUfunction f,
	unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra)
GPU_FUNC_LAUNCH_END(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
	f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);

#endif

