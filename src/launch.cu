// TODO 2) Different socket PORT numbers for HIP (2485) and CUDA (2788)
// TODO 3) FREEZE_ON_START=1 env variable to wait for a start command in socket

#include "easyprof.h"

#include <cxxabi.h>
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

// CUDA/HIP APIs can execute a user callback function, when the corresponding
// stream reaches the point of interest. We use this feature to track kernels
// execution in a simple way.
#ifdef __HIPCC__
static void profilerStartTimer(hipStream_t stream, hipError_t status, void *userData)
#else
static void profilerStartTimer(CUstream stream, CUresult status, void *userData)
#endif
{
	auto launch = *reinterpret_cast<std::tuple<Timer::Launches*, int>*>(userData);

	if (Profiler::get().timer->isTiming())
		Profiler::get().timer->start(stream, launch);
}

#ifdef __HIPCC__
static void profilerStopTimer(hipStream_t stream, hipError_t status, void *userData)
#else
static void profilerStopTimer(CUstream stream, CUresult status, void *userData)
#endif
{
	auto launch = reinterpret_cast<std::tuple<Timer::Launches*, int>*>(userData);

	if (Profiler::get().timer->isTiming())
		Profiler::get().timer->stop(stream, *launch);
	
	delete launch;
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
		// Hack around hipExtModuleLaunchKernel, which is aparently _Z24hipExtModuleLaunchKer@@hip_4.2
		if (sym == "hipExtModuleLaunchKernel")
			sym = "_Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_tjjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j";
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
#ifdef __CUDACC__
		struct CUfunc_st *pFunc = (struct CUfunc_st *)f;
		struct kernel *pKernel = pFunc->kernel;

		int status;    
		char* name = abi::__cxa_demangle(pFunc->name, 0, 0, &status);	
		std::string deviceName = status ? pFunc->name : name;
#else
		std::string deviceName = "unknown_" + std::to_string(Profiler::get().funcs.size());
#endif
		// Get the kernel register count.
		int nregs = 0;
		struct gpuFuncAttributes attrs;
		if (gpuFuncGetAttributes(&attrs, (void*)f) != gpuSuccess)
		{
			fprintf(stderr, "Could not read the number of registers for function \"%s\"\n", deviceName.c_str());
			auto err = gpuGetLastError();
		}
		else
		{
			nregs = attrs.numRegs;
		}

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

	auto& func = it->second;
	auto& name = func->deviceName;

	RetTy result;

	// Start profiling the newly-launched kernel.
	if (Profiler::get().matcher->isMatching(name))
	{
		if (Profiler::get().timer->isTiming())
		{
			auto record_ = Profiler::get().timer->measure(func.get(),
				dim3(gridDimX, gridDimY, gridDimZ),
				dim3(blockDimX, blockDimY, blockDimZ),
				stream);

			// XXX This IS horrible, but we are in the rush on developing bad code, right?
			auto record = new decltype(record_);
			*record = record_;

			// Insert a callback into the same stream before and after the launch,
			// in order to have the time measurement started and stopped.
#ifdef __CUDACC__
			auto err = cuStreamAddCallback(stream, profilerStartTimer, /* userData = */ record, 0);

			// Call the real function.
			result = std::invoke(funcReal, args...);

			err = cuStreamAddCallback(stream, profilerStopTimer, /* userData = */ record, 0);
#else
			// in order to have it to stop the time measurement.
			auto err = hipStreamAddCallback(stream, profilerStartTimer, /* userData = */ record, 0);

			// Call the real function.
			result = std::invoke(funcReal, args...);

			err = hipStreamAddCallback(stream, profilerStopTimer, /* userData = */ record, 0);
#endif
		}
	}
	else
	{
		// Call the real function.
		result = std::invoke(funcReal, args...);
	}

	return result;
}

#if defined(__HIPCC__)

// HIP has multiple different API functions for kernel launching.

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

