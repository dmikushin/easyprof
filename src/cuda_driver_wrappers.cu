#ifdef __CUDACC__

#include "easyprof.h"

std::vector<GPUapiFunc> gpuAPI =
{
	GPU_API_NAME(cuGetProcAddress),

	GPU_API_NAME(cuMemAlloc),
	GPU_API_NAME(cuMemAllocFromPoolAsync),
	GPU_API_NAME(cuMemAllocHost),
	GPU_API_NAME(cuMemAllocManaged),
	GPU_API_NAME(cuMemAllocPitch),

	GPU_API_NAME(cuMemcpy),
	GPU_API_NAME(cuMemcpy2D),
	GPU_API_NAME(cuMemcpy2DAsync),
	GPU_API_NAME(cuMemcpy2DUnaligned),
	GPU_API_NAME(cuMemcpy3D),
	GPU_API_NAME(cuMemcpy3DAsync),
	GPU_API_NAME(cuMemcpy3DPeer),
	GPU_API_NAME(cuMemcpy3DPeerAsync),
	GPU_API_NAME(cuMemcpyAsync),
	GPU_API_NAME(cuMemcpyAtoA),
	GPU_API_NAME(cuMemcpyAtoD),
	GPU_API_NAME(cuMemcpyAtoH),
	GPU_API_NAME(cuMemcpyAtoHAsync),
	GPU_API_NAME(cuMemcpyDtoA),
	GPU_API_NAME(cuMemcpyDtoD),
	GPU_API_NAME(cuMemcpyDtoDAsync),
	GPU_API_NAME(cuMemcpyDtoH),
	GPU_API_NAME(cuMemcpyDtoHAsync),
	GPU_API_NAME(cuMemcpyHtoA),
	GPU_API_NAME(cuMemcpyHtoAAsync),
	GPU_API_NAME(cuMemcpyHtoD),
	GPU_API_NAME(cuMemcpyHtoDAsync),
	GPU_API_NAME(cuMemcpyPeer),
	GPU_API_NAME(cuMemcpyPeerAsync),

	GPU_API_NAME(cuMemsetD16),
	GPU_API_NAME(cuMemsetD16Async),
	GPU_API_NAME(cuMemsetD2D16),
	GPU_API_NAME(cuMemsetD2D16Async),
	GPU_API_NAME(cuMemsetD2D32),
	GPU_API_NAME(cuMemsetD2D32Async),
	GPU_API_NAME(cuMemsetD2D8),
	GPU_API_NAME(cuMemsetD2D8Async),
	GPU_API_NAME(cuMemsetD32),
	GPU_API_NAME(cuMemsetD32Async),
	GPU_API_NAME(cuMemsetD8),
	GPU_API_NAME(cuMemsetD8Async),

	GPU_API_NAME(cuLaunchCooperativeKernel),
	GPU_API_NAME(cuLaunchHostFunc),
	GPU_API_NAME(cuLaunchKernel)
};

#endif // __CUDACC__

