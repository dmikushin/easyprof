// For HIP, we wrap over the __hipRegisterFunction() call, because it's the only way
// to obtain the mapping between the device function pointer and device function name.

#include "easyprof.h"

#include <dlfcn.h>

#define bind_lib(path, lib) \
if (!lib) \
{ \
	lib = dlopen(path, RTLD_LAZY | RTLD_GLOBAL); \
	if (!lib) \
	{ \
		LOG("Error loading %s: %s", path, dlerror()); \
		abort(); \
	} \
}

#define bind_sym(handle, sym, retty, ...) \
typedef retty (*sym##_func_t)(__VA_ARGS__); \
static sym##_func_t sym##_real = nullptr; \
if (!sym##_real) \
{ \
	auto name = #sym; \
	sym##_real = (sym##_func_t)SymbolLoader::get(handle, name); \
	if (!sym##_real) \
	{ \
		LOG("Error loading %s : %s", name, dlerror()); \
		abort(); \
	} \
}

#ifdef __HIPCC__

extern "C"
void __hipRegisterFunction(
	void** vfatCubinHandle,
	const char* hostFun,
	char* deviceFun,
	const char* name,
	int thread_limit,
	uint3* tid,
	uint3* bid,
	dim3* bDim,
	dim3* gDim,
	int* wSize)
{
	bind_lib(LIBGPURT, libgpurt);
	bind_sym(libgpurt, __hipRegisterFunction, void,
		void**, const char*, char*, const char*,
		int, uint3*, uint3*, dim3*, dim3*, int*);

	__hipRegisterFunction_real(
		vfatCubinHandle, hostFun, deviceFun, name,
		thread_limit, tid, bid, bDim, gDim, wSize);

	int status;    
	const char* deviceName = abi::__cxa_demangle(name, 0, 0, &status);
	deviceName = status ? name : deviceName;

	int nregs = 0;
	struct hipFuncAttributes attrs;
	if (hipFuncGetAttributes(&attrs, reinterpret_cast<const void*>(hostFun)) != hipSuccess)
	{
		fprintf(stderr, "Could not read the number of registers for function \"%s\"\n", deviceName);
		auto err = gpuGetLastError();
	}
	else
	{
		nregs = attrs.numRegs;
	}

	Profiler::get().addKernel(f, []()
	{
		return GPUfunction
		{
			deviceFun,
			deviceName,
			nregs
		};
	});
}

#endif // __HIPCC__

extern "C"
gpuError_t gpuModuleGetFunction(gpuFunction_t *function, gpuModule_t module, const char *deviceName)
{
	bind_lib(LIBGPURT, libgpurt);
#ifdef __CUDACC__
	bind_sym(libgpu, cuModuleGetFunction, gpuError_t,
		gpuFunction_t*, gpuModule_t, const char*);
	auto result = cuModuleGetFunction_real(function, module, deviceName);
#else
	bind_sym(libgpurt, hipModuleGetFunction, gpuError_t,
		gpuFunction_t*, gpuModule_t, const char*);
	auto result = hipModuleGetFunction_real(function, module, deviceName);
#endif
	if (result != gpuSuccess)
	{
		const char* errStr;
		cuGetErrorString(result, &errStr);
		fprintf(stderr, "Could not load the function \"%s\" from module %p: \"%s\"\n",
			deviceName, module, errStr);
		return result;
	}

	int nregs = 0;
	result = gpuFuncGetAttribute(&nregs, GPU_FUNC_ATTRIBUTE_NUM_REGS, *function);
	if (result != gpuSuccess)
	{
		const char* errStr;
		cuGetErrorString(result, &errStr);
		fprintf(stderr, "Could not read the number of registers for function \"%s\": \"%s\"\n",
			deviceName, errStr);
		auto err = gpuGetLastError();
	}

	Profiler::get().addKernel(*function, [&]()
	{
		return GPUfunction
		{
			*function,
			deviceName,
			nregs
		};
	});

	return result;
}

