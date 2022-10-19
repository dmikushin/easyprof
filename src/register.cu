#if 0
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

static std::string old_api_name(const std::string& sym)
{
	std::string result = sym;
#if __HIPCC__
	return result.replace(result.find("gpu"), 3, "hip");
#else
	return result.replace(result.find("gpu"), 3, "cuda");
#endif
}

#define bind_sym(handle, sym, retty, ...) \
typedef retty (*sym##_func_t)(__VA_ARGS__); \
static sym##_func_t sym##_real = nullptr; \
if (!sym##_real) \
{ \
	auto name = old_api_name(#sym); \
	sym##_real = (sym##_func_t)SymbolLoader::get(handle, name.c_str()); \
	if (!sym##_real) \
	{ \
		LOG("Error loading %s : %s", name.c_str(), dlerror()); \
		abort(); \
	} \
}

extern "C"
void __gpuRegisterFunction(
	void** vfatCubinHandle,
	const char* hostFun,
	char* deviceFun,
	const char* deviceName,
	int thread_limit,
	uint3* tid,
	uint3* bid,
	dim3* bDim,
	dim3* gDim,
	int* wSize)
{
	bind_lib(LIBGPURT, libgpurt);
	bind_sym(libgpurt, __gpuRegisterFunction, void,
		void**, const char*, char*, const char*,
		int, uint3*, uint3*, dim3*, dim3*, int*);

	__gpuRegisterFunction_real(
		vfatCubinHandle, hostFun, deviceFun, deviceName,
		thread_limit, tid, bid, bDim, gDim, wSize);

	int status;    
	char* name = abi::__cxa_demangle(deviceName, 0, 0, &status);

	auto uint3zero = uint3{};
	auto dim3zero = dim3{};
#if 0	
	cuInit(0);
	CUmodule m;
	if (cuModuleLoadFatBinary(&m, vfatCubinHandle) != CUDA_SUCCESS)
	{
		fprintf(stderr, "Error in cuModuleLoad\n");
	}
	CUfunction f;
	if (cuModuleGetFunction(&f, m, deviceName) != CUDA_SUCCESS)
	{
		fprintf(stderr, "Error in cuModuleGetFunction\n");
	}
#endif	
	Profiler::get().funcs[(void*)hostFun] = std::make_shared<GPUfunction>(GPUfunction
	{
		vfatCubinHandle,
		hostFun,
		deviceFun,
		status ? deviceName : name,
		thread_limit,
		(tid ? *tid : uint3zero),
		(bid ? *bid : uint3zero),
		(bDim ? *bDim : dim3zero),
		(gDim ? *gDim : dim3zero),
		(wSize ? *wSize : 0),
		0 // nregs, not available yet
	});
}
#endif

