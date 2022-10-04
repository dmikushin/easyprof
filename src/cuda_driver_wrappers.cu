#ifdef __CUDACC__

#include "easyprof.h"

std::vector<GPUapiFunc> gpuAPI =
{
	GPU_API_NAME(cuGetProcAddress),

	GPU_API_NAME(cuLaunchCooperativeKernel),
	GPU_API_NAME(cuLaunchHostFunc),
	GPU_API_NAME(cuLaunchKernel)
};

#endif // __CUDACC__

