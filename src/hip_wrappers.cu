#ifdef __HIPCC__

#include "easyprof.h"

std::vector<GPUapiFunc> gpuAPI =
{
	GPU_API_NAME(hipLaunchKernel),
	GPU_API_NAME(hipExtLaunchKernel),
	GPU_API_NAME(hipModuleLaunchKernel),
	GPU_API_NAME(hipExtModuleLaunchKernel)
};

#endif // __CUDACC__

