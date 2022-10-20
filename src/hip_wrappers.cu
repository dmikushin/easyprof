#ifdef __HIPCC__

#include "easyprof.h"

std::vector<GPUapiFunc> gpuAPI =
{
	GPU_API_NAME(hipLaunchKernel),
	GPU_API_NAME(hipExtLaunchKernel),
	GPU_API_NAME(hipModuleLaunchKernel),
	{ "_Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_tjjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j",
		reinterpret_cast<void*>(hipExtModuleLaunchKernel) },
	GPU_API_NAME(hipModuleGetFunction)
};

#endif // __CUDACC__

