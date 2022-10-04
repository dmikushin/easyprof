#include "easyprof.h"

#include <cxxabi.h>
#include <dlfcn.h>
#include <functional>

#define GPU_FUNC_LAUNCH_BEGIN(prefix, __stream, __function, \
	__gridDimX, __gridDimY, __gridDimZ, __blockDimX, __blockDimY, __blockDimZ, __sharedMemBytes, \
	RetTy, name, ...) \
	extern "C" \
	RetTy api_name(name, prefix)(__VA_ARGS__) \
	{ \
		gpuStream_t __s = static_cast<gpuStream_t>(__stream); \
		return gpuFuncLaunch<RetTy>( \
			__dll(str_prefix(prefix)), str_api_name(name, prefix), __s, __function, \
			gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,

#define GPU_FUNC_LAUNCH_END(...) \
			__VA_ARGS__); \
	}

#if defined(__HIPCC__)

template<typename RetTy, typename... Args>
RetTy gpuFuncLaunch(const std::string dll, const std::string name, gpuStream_t stream, CUfunction f, Args... args)
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
		funcReal = (Func)SymbolLoader::get(handle, name.c_str());
		if (!funcReal)
		{
			LOG("Error loading %s : %s", name.c_str(), dlerror());
			abort();
		}
	}

	auto& func = profiler.funcs[(void*)f];
	auto name = func->deviceName;

	// Call the real function.
	auto result = std::invoke(funcReal, args...);

	// Start profiling the newly-launched kernel.
	if (profiler.matcher->isMatching(name))
	{
#if 0
		LOG("%s<<<(%u, %u, %u), (%u, %u, %u), %zu, %p>>> = %d\n",
			name.c_str(), gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
			sharedMem, stream, result);
#endif	
		// Don't do anything else, if kernel launch was not successful.
		if (result != gpuSuccess) return result;
		
		if (profiler.timer->isTiming())
		{
			if (!func->nregs)
			{ 
				// Get the kernel register count.
				struct gpuFuncAttributes attrs;
				if (gpuFuncGetAttributes(&attrs, (void*)f) != gpuSuccess)
				{
					fprintf(stderr, "Could not read the number of registers for function \"%s\"\n", name.c_str());
					gpuGetLastError();
				}
				
				func->nregs = attrs.numRegs;
			}

			profiler.timer->measure(func.get(),
				dim3(gridDim.x, gridDim.y, gridDim.z),
				dim3(blockDim.x, blockDim.y, blockDim.z),
				stream);
		}
	}

	return result;
}

GPU_FUNC_LAUNCH_BEGIN(RuntimeLibraryPrefix, stream, gpuError_t, gpuLaunchKernel,
	const void* f, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, gpuStream_t stream);
GPU_FUNC_LAUNCH_END(f, gridDim, blockDim, args, sharedMem, stream);

GPU_FUNC_LAUNCH_BEGIN(RuntimeLibraryPrefix, stream, gpuError_t, gpuExtLaunchKernel,
	const void* f, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMem, gpuStream_t stream,
	gpuEvent_t startEvent, gpuEvent_t stopEvent, int flags);
GPU_FUNC_LAUNCH_END(f, numBlocks, dimBlocks, args, sharedMem, stream, startEvent, stopEvent, flags);

GPU_FUNC_LAUNCH_BEGIN(RuntimeLibraryPrefix, stream, gpuError_t, gpuModuleLaunchKernel,
	gpuFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
	gpuStream_t stream, void** kernelParams, void** extra);
GPU_FUNC_LAUNCH_END(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
	sharedMemBytes, stream, kernelParams, extra);

#else

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

template<typename RetTy, typename... Args>
RetTy gpuFuncLaunch(
	const std::string dll, const std::string sym,
	gpuStream_t stream, CUfunction f,
	unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes,
	Args... args)
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

	struct CUfunc_st *pFunc = (struct CUfunc_st *)f;
	struct kernel *pKernel = pFunc->kernel;
#if 0
	wrapper.addKernel(pFunc->name, pKernel->module,
		gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
		sharedMemBytes, kernelParams);
#endif
	auto it = profiler.funcs.find(pFunc); // TODO Not pFunc?
	if (it == profiler.funcs.end())
	{
		int status;    
		char* name = abi::__cxa_demangle(pFunc->name, 0, 0, &status);	

		profiler.funcs[(void*)f] = std::make_shared<GPUfunction>(GPUfunction
		{
			/* void* vfatCubinHandle */   pKernel->module,
			/* const char* hostFun; */    pFunc->name,
			/* char* deviceFun; */        pFunc->name,
			/* std::string deviceName; */ status ? pFunc->name : name,
			/* int thread_limit; */       0, // thread_limit is not known
			/* uint3 tid; */              dim3 { gridDimX, gridDimY, gridDimZ },
			/* uint3 bid; */              dim3 { blockDimX, blockDimY, blockDimZ },
			/* dim3 bDim; */              dim3 { gridDimX, gridDimY, gridDimZ },
			/* dim3 gDim; */              dim3 { blockDimX, blockDimY, blockDimZ },
			/* int wSize; */              static_cast<int>(sharedMemBytes),
			/* int nregs; */              0 // nregs, not available yet
		});
	}
	auto& func = it->second;
	auto name = func->deviceName;
	printf("%s\n", name.c_str());

	// Call the real function.
	auto result = std::invoke(funcReal, args...);
/*
	// Start profiling the newly-launched kernel.
	if (profiler.matcher->isMatching(name))
	{
#if 0
		LOG("%s<<<(%u, %u, %u), (%u, %u, %u), %zu, %p>>> = %d\n",
			name.c_str(), gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
			sharedMem, stream, result);
#endif	
		// Don't do anything else, if kernel launch was not successful.
		if (result != gpuSuccess) return result;
		
		if (profiler.timer->isTiming())
		{
			if (!func->nregs)
			{ 
				// Get the kernel register count.
				struct gpuFuncAttributes attrs;
				if (gpuFuncGetAttributes(&attrs, (void*)f) != gpuSuccess)
				{
					fprintf(stderr, "Could not read the number of registers for function \"%s\"\n", name.c_str());
					gpuGetLastError();
				}
				
				func->nregs = attrs.numRegs;
			}

			profiler.timer->measure(func.get(),
				dim3(gridDim.x, gridDim.y, gridDim.z),
				dim3(blockDim.x, blockDim.y, blockDim.z),
				stream);
		}
	}
*/
	return result;
}

GPU_FUNC_LAUNCH_BEGIN(DriverLibraryPrefix, hStream, f,
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
	sharedMemBytes,
	CUresult, LaunchKernel,
	CUfunction f,
	unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra)
GPU_FUNC_LAUNCH_END(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
	sharedMemBytes, hStream, kernelParams, extra);

#endif
