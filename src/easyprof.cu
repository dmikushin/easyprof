#include <assert.h>
#include <cxxabi.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <tuple>

#include "easyprof.h"

void* libdl = nullptr;
void* libgpu = nullptr;
void* libgpurt = nullptr;

static bool operator<(const dim3& v1, const dim3& v2)
{
	auto c1 = reinterpret_cast<const char*>(&v1);
	auto c2 = reinterpret_cast<const char*>(&v2);
	return memcmp(c1, c2, sizeof(dim3)) > 0;
}

void Profiler::rotateArchive()
{
	archive.emplace_back();
	launches = &archive.back();
	launches->reserve(1024 * 1024);
}

Profiler::Profiler()
{
	// Allocate the first slice of space for profiling data.
	rotateArchive();
}

Profiler& Profiler::get()
{
	static Profiler profiler;
	return profiler;
}

Launch* Profiler::start(const void* deviceFun,
	const dim3& numBlocks, const dim3& dimBlocks,
	unsigned int sharedMemBytes, gpuStream_t stream)
{
	if (launches->size() == launches->capacity())
		rotateArchive();
	
	Timestamp begin {}, end {};
	launches->push_back(Launch
	{
		deviceFun,
		begin, end,
		numBlocks, dimBlocks,
		sharedMemBytes, stream
	});

	auto launch = &launches->at(launches->size() - 1);
	
	// CUDA/HIP APIs can execute a user callback function, when the corresponding
	// stream reaches the point of interest. We use this feature to track kernels
	// execution in a simple way.
#ifdef __CUDACC__
	auto err = cuStreamAddCallback(stream, [](CUstream stream, CUresult status, void *userData)
#else
	auto err = hipStreamAddCallback(stream, [](hipStream_t stream, hipError_t status, void *userData)
#endif
	{
		auto& launch = *reinterpret_cast<Launch*>(userData);
		launch.begin = std::chrono::high_resolution_clock::now();
	},
	launch, 0);
	
	return launch;
}

void Profiler::stop(gpuStream_t stream, Launch* launch)
{
#ifdef __CUDACC__
	auto err = cuStreamAddCallback(stream, [](CUstream stream, CUresult status, void *userData)
#else
	auto err = hipStreamAddCallback(stream, [](hipStream_t stream, hipError_t status, void *userData)
#endif
	{
		auto& launch = *reinterpret_cast<Launch*>(userData);
		launch.end = std::chrono::high_resolution_clock::now();
	},
	launch, 0);
}

Profiler::~Profiler()
{
	size_t count = 0;
	for (const auto& launches : archive)
		count += launches.size();

	printf("Processing %zu recordings for %zu kernels, this may take a while\n",
		count, funcs.size());

	std::map<
		std::string, // kernel name
		std::pair<
			unsigned int, // the number of registers for the kernel
			std::map<
				std::tuple<dim3, dim3, unsigned int>, // grid dim, block dim, shared memory
				std::tuple<
					double, double, double, // min/max/average time
					unsigned int // number of calls
				>
			>
		>
	> results;

	// Demangle names of kernels.
	{
		int i = 0;
		for (auto& [_, func] : funcs)
		{
			auto& name = func.deviceName;
			if (name == "")
				name = std::to_string(i++);
			else
			{
				int status;    
				char* nameDemangled = abi::__cxa_demangle(name.c_str(), 0, 0, &status);
				if (status == 0)
					name = nameDemangled;	
			}
		}
	}

	// Accumulate the results.
	bool notFullyCaptured = false;
	for (auto& [_, func] : funcs)
	{
		const auto& deviceFun = func.deviceFun;
		const auto& deviceName = func.deviceName;
		const auto nregs = func.nregs;

		auto& result = results[deviceName];
		result.first = nregs;
		
		// TODO Get the number of registers for the kernel here.
			
		for (const auto& launches : archive)
		{
			for (const auto& launch : launches)
			{
				if (launch.deviceFun != deviceFun) continue;
			
				const auto& begin = launch.begin;
				const auto& end = launch.end;
				
				// Ignore launches, which are not fully captured.
				std::chrono::time_point<std::chrono::high_resolution_clock> zero {};
				if (end == zero)
				{
					notFullyCaptured = true;
					continue;
				}
				auto duration = std::chrono::duration<double, std::micro>{end - begin}.count();
				if (duration <= 0.0)
				{
					notFullyCaptured = true;
					continue;
				}

				const auto& numBlocks = launch.numBlocks;
				const auto& dimBlocks = launch.dimBlocks;
				const auto& sharedMemBytes = launch.sharedMemBytes;

				auto& stats = result.second[std::make_tuple(numBlocks, dimBlocks, sharedMemBytes)];
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
				
				assert(min > 0.0);
				assert(max > 0.0);
				
				avg += duration;
				ncalls++;
			} 
		}
	}
	
	if (notFullyCaptured)
	{
		fprintf(stderr, "Not all kernels launches were captured correctly,"
			" the app could have GPU errors or is interrupted\n");
	}
		
	// Conclude and report the results.
	for (auto& result : results)
	{
		const auto& name = result.first;			
		const auto nregisters = result.second.first;
		
		// Report results only for the kernels that are launched at least once.
		if (!result.second.second.size()) continue;

		// TODO If the name is long, maybe shorten it to the last part after ::
		std::cout << "\"" << name << "\"" << " (" << nregisters << " registers)" << std::endl;
		
		for (auto& grid : result.second.second)
		{
			const dim3& numBlocks = std::get<0>(grid.first);
			const dim3& dimBlocks = std::get<1>(grid.first);
			const auto& sharedMemBytes = std::get<2>(grid.first);
			
			const auto min = std::get<0>(grid.second);
			const auto max = std::get<1>(grid.second);
			auto& avg = std::get<2>(grid.second);
			const unsigned int ncalls = std::get<3>(grid.second);
			avg /= ncalls;
			
			std::cout << ncalls << " x <<<(" << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << "), (" <<
				dimBlocks.x << ", " << dimBlocks.y << ", " << dimBlocks.z << ")" <<
				(sharedMemBytes ? ", " + std::to_string(sharedMemBytes) : "") << ">>> " <<
				"min = " << min << "ms, max = " << max << "ms, avg = " << avg << "ms" << std::endl;
		}
	}

	// Look on how many different streams IDs the kernel was executed.
	std::map<const void*, std::map<gpuStream_t, unsigned int> > kernels_streams;
	for (const auto& launches : archive)
		for (const auto& launch : launches)
		{
			const auto& stream = launch.stream;
			const auto& deviceFun = launch.deviceFun;

			kernels_streams[deviceFun][stream]++;
		}

	for (auto& [_, func] : funcs)
	{
		const auto& deviceFun = func.deviceFun;
		const auto& deviceName = func.deviceName;

		const auto& streams = kernels_streams[deviceFun];
		if (streams.size() == 0) continue;

		// If kernel is launched only on the default (0) stream, do not print anything.
		if ((streams.size() == 1) && (streams.at(static_cast<gpuStream_t>(0)) != 0)) continue;

		// TODO If the name is long, maybe shorten it to the last part after ::
		std::cout << "\"" << deviceName << "\"" << " non-default streams:";
		for (auto & [id, count] : streams)
			std::cout << " [ stream = " << id << " : ncalls = " << count << " ]";
		std::cout << std::endl;
	}
}

std::map<std::string, void*> dlls;

const char* __dll(const char* prefix)
{
	std::string sprefix = prefix;
#if defined(__CUDACC__)
	if (sprefix == "cu") return LIBGPU;
	if (sprefix == "cuda") return LIBGPURT;
#elif defined(__HIPCC__)
	if (sprefix == "hip") return LIBGPURT;
#endif
	return "";
}


