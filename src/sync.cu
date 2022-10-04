#include "easyprof.h"

#include <dlfcn.h>
#include <functional>

#define GPU_FUNC_SYNC_BEGIN(prefix, __stream, RetTy, name, ...) \
	extern "C" \
	RetTy api_name(name, prefix)(__VA_ARGS__) \
	{ \
		gpuStream_t __s = static_cast<gpuStream_t>(__stream); \
		return gpuFuncSynchronize<RetTy>( \
			__dll(str_prefix(prefix)), str_api_name(name, prefix), __s,

#define GPU_FUNC_SYNC_END(...) \
			__VA_ARGS__); \
	}

template<typename RetTy, typename... Args>
RetTy gpuFuncSynchronize(const std::string dll, const std::string name, gpuStream_t stream, Args... args)
{
	void* handle = nullptr;
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

	// This function implicitly synchronizes its working stream,
	// so we must synchronize any outstanding operations in our profiler
	// for it as well.
	if (profiler.timer->isTiming())
	{
#ifdef __HIPCC__
		gpuStreamSynchronize(stream);
#else
		// In CUDA, cudaDeviceSynchronize calls cuCtxSynchronize
		cuCtxSynchronize();
#endif
	}

	// Call the real function.
	return std::invoke(funcReal, args...);
}

GPU_FUNC_SYNC_BEGIN(DriverLibraryPrefix, 0, CUresult, MemcpyDtoH,
	void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
GPU_FUNC_SYNC_END(dstHost, srcDevice, ByteCount);

GPU_FUNC_SYNC_BEGIN(DriverLibraryPrefix, hStream, CUresult, MemcpyDtoHAsync,
	void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
GPU_FUNC_SYNC_END(dstHost, srcDevice, ByteCount, hStream);

#ifdef __HIPCC__

GPU_FUNC_SYNC_BEGIN(RuntimeLibraryPrefix, 0, gpuError_t, gpuDeviceSynchronize)
GPU_FUNC_SYNC_END();

GPU_FUNC_SYNC_BEGIN(RuntimeLibraryPrefix, stream, gpuError_t, gpuStreamSynchronize, gpuStream_t stream)
GPU_FUNC_SYNC_END(stream);

GPU_FUNC_SYNC_BEGIN(RuntimeLibraryPrefix, stream, gpuError_t, gpuMemcpyAsync,
	void *dst, const void *src, size_t count, enum gpuMemcpyKind kind, gpuStream_t stream)
GPU_FUNC_SYNC_END(dst, src, count, kind, stream);

GPU_FUNC_SYNC_BEGIN(RuntimeLibraryPrefix, 0, gpuError_t, gpuMemcpy,
	void *dst, const void *src, size_t count, enum gpuMemcpyKind kind)
GPU_FUNC_SYNC_END(dst, src, count, kind);

#endif // __HIPCC__

