#include <algorithm>
#include <chrono>
#include <cxxabi.h>
#include <dlfcn.h>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <regex>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <tuple>
#include <vector>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
using gpuError_t = hipError_t;
using gpuStream_t = hipStream_t;
static const auto gpuSuccess = hipSuccess;
#define __gpuRegisterFunction __hipRegisterFunction
#define gpuFuncAttributes hipFuncAttributes
#define gpuFuncGetAttributes(...) hipFuncGetAttributes(__VA_ARGS__)
#define gpuGetLastError(...) hipGetLastError(__VA_ARGS__)
#define gpuLaunchKernel(...) hipLaunchKernel(__VA_ARGS__)
#define gpuStreamSynchronize(...) hipStreamSynchronize(__VA_ARGS__)
#define gpuDeviceSynchronize() hipDeviceSynchronize()
#define LIBGPURT "/opt/rocm/hip/lib/libamdhip64.so"
#else
#include <cuda_runtime.h>
using gpuError_t = cudaError_t;
using gpuStream_t = cudaStream_t;
static const auto gpuSuccess = cudaSuccess;
#define __gpuRegisterFunction __cudaRegisterFunction
#define gpuFuncAttributes cudaFuncAttributes
#define gpuFuncGetAttributes(...) cudaFuncGetAttributes(__VA_ARGS__)
#define gpuGetLastError(...) cudaGetLastError(__VA_ARGS__)
#define gpuLaunchKernel(...) cudaLaunchKernel(__VA_ARGS__)
#define gpuStreamSynchronize(...) cudaStreamSynchronize(__VA_ARGS__)
#define gpuDeviceSynchronize() cudaDeviceSynchronize()
#define LIBGPURT "/usr/local/cuda/lib64/libcudart.so"
#endif

#define LOG(...) printf(__VA_ARGS__)

static void* libdl = nullptr;
static void* libgpurt = nullptr;

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

static std::string api_name(const std::string& sym)
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
	auto name = api_name(#sym); \
	sym##_real = (sym##_func_t)dlsym(handle, name.c_str()); \
	if (!sym##_real) \
	{ \
		LOG("Error loading %s : %s", #sym, dlerror()); \
		abort(); \
	} \
}

struct GPUfunction
{
	void* vfatCubinHandle;
	const char* hostFun;
	char* deviceFun;
	std::string deviceName;
	int thread_limit;
	uint3 tid;
	uint3 bid;
	dim3 bDim;
	dim3 gDim;
	int wSize;
	int nregs;
};

class Matcher;
class Timer;

// Maintaining the proper order of destruction.
struct Profiler
{
	std::map<void*, std::shared_ptr<GPUfunction>> funcs;

	Matcher* matcher;

	// TODO must support threads.
	Timer* timer;
	
	Profiler();
	~Profiler();
};

static Profiler profiler;

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
#if 0
#define VAL_OR_NIL(ptr, prop) (ptr ? ((ptr)->prop) : 0)
	LOG("__gpuRegisterFunction(\"%s\", %p, %p, %p, %d, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %d)\n",
		deviceName, vfatCubinHandle, hostFun, deviceFun, thread_limit,
		VAL_OR_NIL(tid, x), VAL_OR_NIL(tid, y), VAL_OR_NIL(tid, z),
		VAL_OR_NIL(bid, x), VAL_OR_NIL(bid, y), VAL_OR_NIL(bid, z),
		VAL_OR_NIL(bDim, x), VAL_OR_NIL(bDim, y), VAL_OR_NIL(bDim, z),
		VAL_OR_NIL(gDim, x), VAL_OR_NIL(gDim, y), VAL_OR_NIL(gDim, z),
		wSize ? *wSize : 0);
#endif
	int status;    
	char* name = abi::__cxa_demangle(deviceName, 0, 0, &status);

	auto uint3zero = uint3{};
	auto dim3zero = dim3{};
	profiler.funcs[(void*)hostFun] = std::make_shared<GPUfunction>(GPUfunction
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

class Matcher
{
	const std::map<void*, std::shared_ptr<GPUfunction>>& funcs;

	std::string pattern;

public :

	bool isMatching(const std::string& name)
	{
		if (pattern == "") return true;
		
		std::smatch m;
		const std::regex r(pattern);
		if (std::regex_match(name, m, r))
			return true;
		
		return false;
	}

	Matcher(const std::map<void*, std::shared_ptr<GPUfunction>>& funcs_) : funcs(funcs_)
	{
		const char* cpattern = getenv("PROFILE_REGEX");
		if (cpattern) pattern = cpattern;
	}
};

static bool operator<(const dim3& v1, const dim3& v2)
{
	auto c1 = reinterpret_cast<const char*>(&v1);
	auto c2 = reinterpret_cast<const char*>(&v2);
	return memcmp(c1, c2, sizeof(dim3)) >= 0;
}

class Timer
{
	const std::map<void*, std::shared_ptr<GPUfunction>>& funcs;

	bool timing = false;
	
	// Enforce stream synchronization, when measuring the kernel
	// execution time.
	bool synced = false;

	using launches =
		std::vector<
			std::tuple<
				std::chrono::time_point<std::chrono::high_resolution_clock>, // time begin
				std::chrono::time_point<std::chrono::high_resolution_clock>, // time end
				dim3, dim3, // gridDim, blockDim
				int // synchronization group index
			>
		>;

	std::map<
		gpuStream_t, // for each stream
		std::map<
			std::string, // for each kernel name
			std::tuple<
				unsigned int, // the number of kernels to sync
				const GPUfunction*, // the corresponding registered function
				launches, // launches for the current time interval
				std::shared_ptr<std::list<launches> > // archived launches from the previous live intervals
			>
		>
	> kernels;
	
	// Archived launches from the previous live intervals,
	// which is to be filled by a long-running app with many kernel launches.
	std::map<GPUfunction*, std::shared_ptr<std::list<launches> > > archive;
	
	int sync_group_index = 0;
	
public :

	bool isTiming() { return timing; }
	
	Timer(const std::map<void*, std::shared_ptr<GPUfunction>>& funcs_) : funcs(funcs_)
	{
		const char* ctiming = getenv("PROFILE_TIME");
		if (ctiming)
		{
			std::string stiming = ctiming;
			std::transform(stiming.begin(), stiming.end(), stiming.begin(),
				[](unsigned char c){ return std::tolower(c); });
			if (stiming == "synced")
			{
				timing = true;
				synced = true;
			}
			else
			{
				timing = atoi(ctiming);
				synced = false;
			}
		}
	}

	void measure(const GPUfunction* func_,
		const dim3& gridDim, const dim3& blockDim, gpuStream_t stream)
	{
		auto& kernel = kernels[stream][func_->deviceName];
		auto& launches = std::get<2>(kernel);
		if ((launches.size() == 0) || (launches.size() == launches.capacity()))
		{
			// Assign the corresponding registered function.
			auto& func = std::get<1>(kernel);
			func = func_;
		
			// Reserve a lot of memory in advance to make re-allocations
			// less disruptive.
			launches.reserve(launches.size() + 1024 * 1024);
		}
	
		auto begin = std::chrono::high_resolution_clock::now();
		std::chrono::time_point<std::chrono::high_resolution_clock> end;
	
		if (synced)
		{
			gpuError_t status = gpuStreamSynchronize(stream);
			
			// TODO If status is bad, forward it to the next user call
			// of gpuStreamSynchronize().
			
			end = std::chrono::high_resolution_clock::now();
		}
		else
		{
			// In asynchronous mode, we maintain how many latest
			// kernel launches we need to synchronize.
			auto& to_sync = std::get<0>(kernel);
			to_sync++;
		}
		
		launches.push_back(std::make_tuple(begin, end, gridDim, blockDim, 0));
	}

	void sync(gpuStream_t stream)
	{
		if (synced) return;

		auto end = std::chrono::high_resolution_clock::now();
		
		// Synchonize all kernels of stream, which are not yet synchronized.
		for (auto& pair : kernels[stream])
		{
			const auto& name = pair.first;
			auto& kernel = pair.second;
			
			auto& launches = std::get<2>(kernel);
			auto& to_sync = std::get<0>(kernel);
			to_sync = std::min(to_sync, static_cast<unsigned int>(launches.size()));
			for (int i = 0, ii = launches.size() - to_sync; i < to_sync; i++)
			{
				// Set the ending timestamp and the synchronization order index.
				std::get<1>(launches[ii]) = end;
				std::get<4>(launches[ii]) = sync_group_index;
			}

			// Reset the unsynced counter.
			to_sync = 0;
		}
		
		sync_group_index++;
	}
	
	void sync()
	{
		// Sync all outstanding streams.
		for (auto& pair : kernels)
			sync(pair.first);
	}
	
	~Timer()
	{
		std::map<
			std::string, // kernel name
			std::pair<
				unsigned int, // the number of registers for the kernel
				std::map<
					std::pair<dim3, dim3>, // grid dim, block dim
					std::tuple<
						double, double, double, // min/max/average time
						unsigned int // number of calls
					>
				>
			>
		> results;

		// Accumulate the results.		
		for (auto& pair : kernels)
		{
			for (auto& kernel : pair.second)
			{
				const auto& name = kernel.first;

				const auto& unsynced = std::get<0>(kernel.second);
				if (unsynced)
				{
					fprintf(stderr, "Error: kernel \"%s\" contains %d unsynced launches!\n",
						name.c_str(), unsynced);
				}

				// Retrieve the number of registers.
				auto& func = std::get<1>(kernel.second);
				unsigned int nregisters = func->nregs;
				
				const auto& timings = std::get<2>(kernel.second);
				for (const auto& timing : timings)
				{
					const auto& begin = std::get<0>(timing);
					const auto& end = std::get<1>(timing);
					auto duration = std::chrono::duration<double, std::micro>{end - begin}.count();
					const auto& gridDim = std::get<2>(timing);
					const auto& blockDim = std::get<3>(timing);
					
					results[name].first = nregisters;
					auto& stats = results[name].second[std::make_pair(gridDim, blockDim)];
					auto& min = std::get<0>(stats);
					auto& max = std::get<1>(stats);
					auto& avg = std::get<2>(stats);
					auto& ncalls = std::get<3>(stats);
					
					if (ncalls)
					{
						min = std::min(min, duration);
						max = std::max(max, duration);
					}
					else
					{
						min = duration;
						max = duration;
					}
					
					avg += duration;
					ncalls++;
				}
			}
		}

		// Conclude and report the results.
		for (auto& result : results)
		{
			const auto& name = result.first;			
			const auto nregisters = result.second.first;
	
			// TODO If the name is long, shorten it to the last part after ::
			std::cout << name << " (" << nregisters << " registers)" << std::endl;
			
			for (auto grid : result.second.second)
			{
				const dim3& gridDim = grid.first.first;
				const dim3& blockDim = grid.first.second;
				
				const auto min = std::get<0>(grid.second);
				const auto max = std::get<1>(grid.second);
				auto& avg = std::get<2>(grid.second);
				const unsigned int ncalls = std::get<3>(grid.second);
				avg /= ncalls;
				
				std::cout << ncalls << " x <<<(" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << "), (" <<
					blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")>>> " <<
					"min = " << min << "ms, max = " << max << "ms, avg = " << avg << "ms" << std::endl;
			}
		}
	}
};

Profiler::Profiler()
{
	matcher = new Matcher(funcs);
	timer = new Timer(funcs);
}

Profiler::~Profiler()
{
	delete matcher;
	delete timer;
};

extern "C"
gpuError_t gpuLaunchKernel(
	const void* f,
	dim3 gridDim,
	dim3 blockDim,
	void** args,
	size_t sharedMem,
	gpuStream_t stream)
{
	bind_lib(LIBGPURT, libgpurt);
	bind_sym(libgpurt, gpuLaunchKernel, gpuError_t,
		const void*, dim3, dim3, void**, size_t, gpuStream_t);

	gpuError_t result = gpuLaunchKernel_real(f, gridDim, blockDim, args, sharedMem, stream);
	auto& func = profiler.funcs[(void*)f];
	auto name = func->deviceName;
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

extern "C"
gpuError_t gpuStreamSynchronize(
	gpuStream_t stream)
{
	bind_lib(LIBGPURT, libgpurt);
	bind_sym(libgpurt, gpuStreamSynchronize, gpuError_t, gpuStream_t);

	gpuError_t result = gpuStreamSynchronize_real(stream);
	if (result != gpuSuccess) return result;
		
	if (profiler.timer->isTiming())
	{
		profiler.timer->sync(stream);
	}
	
	return result;
}

extern "C"
gpuError_t gpuDeviceSynchronize()
{
	bind_lib(LIBGPURT, libgpurt);
	bind_sym(libgpurt, gpuDeviceSynchronize, gpuError_t);

	gpuError_t result = gpuDeviceSynchronize_real();
	if (result != gpuSuccess) return result;
		
	if (profiler.timer->isTiming())
	{
		profiler.timer->sync();
	}
	
	return result;
}

