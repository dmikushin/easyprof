#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <regex>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <tuple>
#include <vector>

#define LIBDL "libdl.so"
#define LIBCUDART "/usr/local/cuda/lib64/libcudart.so"

#define LOG(...) printf(__VA_ARGS__)

static void* libdl = nullptr;
static void* libcudart = nullptr;

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
	sym##_real = (sym##_func_t)dlsym(handle, #sym); \
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
};

std::map<void*, GPUfunction> funcs;

extern "C"
void CUDARTAPI __cudaRegisterFunction(
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
	bind_lib(LIBCUDART, libcudart);
	bind_sym(libcudart, __cudaRegisterFunction, void,
		void**, const char*, char*, const char*,
		int, uint3*, uint3*, dim3*, dim3*, int*);

	__cudaRegisterFunction_real(
		vfatCubinHandle, hostFun, deviceFun, deviceName,
		thread_limit, tid, bid, bDim, gDim, wSize);
#if 0
#define VAL_OR_NIL(ptr, prop) (ptr ? ((ptr)->prop) : 0)
	LOG("__cudaRegisterFunction(\"%s\", %p, %p, %p, %d, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %d)\n",
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
	funcs[(void*)hostFun] =
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
		(wSize ? *wSize : 0)
	};
}

class Matcher
{
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

	Matcher()
	{
		const char* cpattern = getenv("PROFILE_REGEX");
		if (cpattern) pattern = cpattern;
	}
};

static Matcher matcher;

static bool operator<(const dim3& v1, const dim3& v2)
{
	auto c1 = reinterpret_cast<const char*>(&v1);
	auto c2 = reinterpret_cast<const char*>(&v2);
	return memcmp(c1, c2, sizeof(dim3)) >= 0;
}

class Timer
{
	bool timing = false;
	
	// Enforce stream synchronization, when measuring the kernel
	// execution time.
	bool synced = false;

	std::map<
		cudaStream_t, // for each stream
		std::map<
			std::string, // for each kernel name
			std::pair<
				unsigned int, // how many unsynced kernels
				std::vector<
					std::tuple<
						std::chrono::time_point<std::chrono::high_resolution_clock>, // time begin
						std::chrono::time_point<std::chrono::high_resolution_clock>, // time end
						dim3, dim3, // gridDim, blockDim
						int // synchronization group index
					>
				>
			>
		>
	> kernels;
	
	int sync_group_index = 0;
	
public :

	bool isTiming() { return timing; }
	
	Timer()
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

	void measure(const std::string& name,
		const dim3& gridDim, const dim3& blockDim, cudaStream_t stream)
	{
		auto kernel = kernels[stream][name];
		if ((kernel.second.size() == 0) || (kernel.second.size() == kernel.second.capacity()))
		{
			// Reserve a lot of memory in advance to make re-allocations
			// less disruptive.
			kernel.second.reserve(kernel.second.size() + 1024 * 1024);
		}
	
		auto begin = std::chrono::high_resolution_clock::now();
		std::chrono::time_point<std::chrono::high_resolution_clock> end;
		bool is_synced = false;
	
		if (synced)
		{
			cudaError_t status = cudaStreamSynchronize(stream);
			
			// TODO If status is bad, forward it to the next user call
			// of cudaStreamSynchronize().
			
			end = std::chrono::high_resolution_clock::now();
		}
		else
		{
			// In asynchronous mode, we maintain how many latest
			// kernel launches we need to synchronize.
			kernel.first++;
		}
		
		kernel.second.push_back(std::make_tuple(begin, end, gridDim, blockDim, 0));
	}

	void sync(cudaStream_t stream)
	{
		if (synced) return;

		auto end = std::chrono::high_resolution_clock::now();
		
		// Synchonize all kernels of stream, which are not yet synchronized.
		for (auto& pair : kernels[stream])
		{
			const auto& name = pair.first;
			auto& kernel = pair.second;
			
			auto& to_sync = kernel.first;
			to_sync = std::min(to_sync, static_cast<unsigned int>(kernel.second.size()));
			for (int i = 0, ii = kernel.second.size() - 1 - to_sync; i < to_sync; i++)
			{
				// Set the ending timestamp and the synchronization order index.
				std::get<1>(kernel.second[ii]) = end;
				std::get<4>(kernel.second[ii]) = sync_group_index;
			}

			// Reset the unsynced counter.
			to_sync = kernel.first;
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

				const auto& unsynced = kernel.second.first;
				if (unsynced)
				{
					fprintf(stderr, "Error: kernel \"%s\" contains %d unsycned launches!\n",
						name.c_str(), unsynced);
				}
				
				const auto& timings = kernel.second.second;
				for (const auto& timing : timings)
				{
					const auto& begin = std::get<0>(timing);
					const auto& end = std::get<1>(timing);
					auto duration = std::chrono::duration<double, std::micro>{end - begin}.count();
					const auto& gridDim = std::get<2>(timing);
					const auto& blockDim = std::get<3>(timing);
					
					// TODO Retrieve the number of registers.
					unsigned int nregisters = 0;
					
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
					"min = " << min << ", max = " << max << ", avg = " << avg << std::endl;
			}
		}
	}
};

// TODO must support threads.
static Timer timer;

extern "C"
cudaError_t cudaLaunchKernel(
	const void* func,
	dim3 gridDim,
	dim3 blockDim,
	void** args,
	size_t sharedMem,
	cudaStream_t stream)
{
	bind_lib(LIBCUDART, libcudart);
	bind_sym(libcudart, cudaLaunchKernel, cudaError_t,
		const void*, dim3, dim3, void**, size_t, cudaStream_t);

	cudaError_t result = cudaLaunchKernel_real(func, gridDim, blockDim, args, sharedMem, stream);
	auto name = funcs[(void*)func].deviceName;
	if (matcher.isMatching(name))
	{
#if 0
		LOG("%s<<<(%u, %u, %u), (%u, %u, %u), %zu, %p>>> = %d\n",
			name.c_str(), gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
			sharedMem, stream, result);
#endif	
		// Don't do anything else, if kernel launch was not successful.
		if (result != cudaSuccess) return result;
		
		if (timer.isTiming())
		{
			timer.measure(name,
				dim3(gridDim.x, gridDim.y, gridDim.z),
				dim3(blockDim.x, blockDim.y, blockDim.z),
				stream);
		}
	}

	return result;
}

extern "C"
cudaError_t cudaStreamSynchronize(
	cudaStream_t stream)
{
	bind_lib(LIBCUDART, libcudart);
	bind_sym(libcudart, cudaStreamSynchronize, cudaError_t, cudaStream_t);

	cudaError_t result = cudaStreamSynchronize_real(stream);
	if (result != cudaSuccess) return result;
		
	if (timer.isTiming())
	{
		timer.sync(stream);
	}
	
	return result;
}

extern "C"
cudaError_t cudaDeviceSynchronize()
{
	bind_lib(LIBCUDART, libcudart);
	bind_sym(libcudart, cudaDeviceSynchronize, cudaError_t);

	cudaError_t result = cudaDeviceSynchronize_real();
	if (result != cudaSuccess) return result;
		
	if (timer.isTiming())
	{
		timer.sync();
	}
	
	return result;
}

