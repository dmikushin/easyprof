#include "easyprof.h"

#include <dlfcn.h>

// Perhaps, we can live without dlsym() wrapping in case of HIP,
// because no dynamic API load is happening, as in case of CUDA's cuGetProcAddress()
// TODO Also, in case of CUDA we should stop wrapping after the API
// is loaded, because wrapping everything is very expensive.
//#define DLSYM_WRAPPER

// Intercept dynamically-loaded API calls, such as in the case
// of statically-linked cudart.
// https://stackoverflow.com/questions/15599026/how-can-i-intercept-dlsym-calls-using-ld-preload/18825060#18825060

void* SymbolLoader::get(void* lib, const char* sym)
{
	static SymbolLoader symLoad;
	return symLoad.dlsym_real(lib, sym);
}

SymbolLoader::SymbolLoader()
{
	dlsym_real = reinterpret_cast<dlsym_t>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
	if (!dlsym_real)
	{
		fprintf(stderr, "Error mapping dlsym symbol: \"%s\"\n", dlerror());
		exit(-1);
	}
#ifdef DLSYM_WRAPPER
	else if (dlsym_real == dlsym)
	{
		fprintf(stderr, "The real dlsym() cannot be equal to the dlsym() wrapper\n");
		exit(-1);
	}
#endif
}

// Perhaps, we can live without dlsym() wrapping in case of HIP,
// because no dynamic API load is happening, as in case of CUDA's cuGetProcAddress()
#ifdef DLSYM_WRAPPER

static int __strcmp(const char* s1, const char* s2)
{
	for (int i = 0; ; i++)
	{
		if (s1[i] != s2[i])
			return s1[i] > s2[i] ? 1 : -1;
		if (s1[i] == '\0')
			break;
	}

	return 0;
}

extern "C" 
void* dlsym(void *handle, const char *name) __THROW
{
	if (!__strcmp(name, "dlsym"))
		return (void*)dlsym;

	for (auto& gpuAPIFunc : gpuAPI)
	{
		if (!__strcmp(name, gpuAPIFunc.name))
			return gpuAPIFunc.ptr;
	}

	void* result = SymbolLoader::get(handle, name);
#if 0
	LOG("dlsym(%p, %s) = %p\n", handle, name, result);
#endif
	return result;
}

#endif // DLSYM_WRAPPER

#ifdef __CUDACC__

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
	sym##_real = (sym##_func_t)SymbolLoader::get(handle, #sym); \
	if (!sym##_real) \
	{ \
		LOG("Error loading %s : %s", #sym, dlerror()); \
		abort(); \
	} \
}

extern "C"
CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags)
{
	bind_lib(LIBGPU, libgpu);
	bind_sym(libgpu, cuGetProcAddress, CUresult,
		const char*, void**, int, cuuint64_t);

	auto result = cuGetProcAddress_real(symbol, pfn, cudaVersion, flags);

	for (auto& gpuAPIFunc : gpuAPI)
	{
		if (!__strcmp(symbol, gpuAPIFunc.name))
			*pfn = gpuAPIFunc.ptr;
	}

	return result;
}

#endif

