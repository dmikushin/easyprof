// For HIP, we wrap over the __hipRegisterFunction() call, because it's the only way
// to obtain the mapping between the device function pointer and device function name.

#ifdef __HIPCC__

#include "easyprof.h"

#include <cxxabi.h>
#include <dlfcn.h>

#define bind_lib(path, lib) \
if (!lib) \
{ \
	lib = dlopen(path, RTLD_NOW | RTLD_GLOBAL); \
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
	
	unsigned int sharedMemBytes = 0;
	Profiler::get().funcs.emplace(reinterpret_cast<const void*>(hostFun),
		std::make_shared<GPUfunction>(GPUfunction
	{
		/* std::string deviceName; */      deviceName,
		/* char* deviceFun; */             deviceFun,
		/* void* module */                 vfatCubinHandle,
		/* unsigned int sharedMemBytes; */ sharedMemBytes,
		/* int nregs; */                   nregs
	}));
}

#endif // __HIPCC__

