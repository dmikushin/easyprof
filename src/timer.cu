#include "easyprof.h"

#include <algorithm>
#include <assert.h>
#include <iostream>

static bool operator<(const dim3& v1, const dim3& v2)
{
	auto c1 = reinterpret_cast<const char*>(&v1);
	auto c2 = reinterpret_cast<const char*>(&v2);
	return memcmp(c1, c2, sizeof(dim3)) > 0;
}

bool Timer::isTiming() { return timing; }

Timer::Timer(const std::map<const void*, std::shared_ptr<GPUfunction>>& funcs_) : funcs(funcs_)
{
	const char* ctiming = getenv("PROFILE_TIME");
	if (ctiming)
	{
		std::string stiming = ctiming;
		std::transform(stiming.begin(), stiming.end(), stiming.begin(),
			[](unsigned char c){ return std::tolower(c); });
		timing = atoi(ctiming);
	}
}

std::tuple<Timer::Launches*, int> Timer::measure(const GPUfunction* func_,
	const dim3& gridDim, const dim3& blockDim, gpuStream_t stream)
{
	auto& kernel = kernels[stream][func_->deviceName];
	auto& launches = std::get<1>(kernel);
	if ((launches.size() == 0) || (launches.size() == launches.capacity()))
	{
		// Assign the corresponding registered function.
		auto& func = std::get<0>(kernel);
		func = func_;
	
		// Reserve a lot of memory in advance to make re-allocations
		// less disruptive.
		launches.reserve(launches.size() + 1024 * 1024);
	}

	std::chrono::time_point<std::chrono::high_resolution_clock> begin, end {};

	launches.emplace_back(std::make_tuple(begin, end, gridDim, blockDim));
	return std::make_tuple(&launches, launches.size() - 1);
}

void Timer::start(gpuStream_t stream, std::tuple<Timer::Launches*, int>& launch)
{
	auto& launches = *std::get<0>(launch);
	auto& i = std::get<1>(launch);
	
	auto begin = std::chrono::high_resolution_clock::now();
	std::get<0>(launches[i]) = begin;
}

void Timer::stop(gpuStream_t stream, std::tuple<Timer::Launches*, int>& launch)
{
	auto& launches = *std::get<0>(launch);
	auto& i = std::get<1>(launch);
	
	auto end = std::chrono::high_resolution_clock::now();
	std::get<1>(launches[i]) = end;
}

Timer::~Timer()
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

			// Retrieve the number of registers.
			auto& func = std::get<0>(kernel.second);
			unsigned int nregisters = func->nregs;
			
			const auto& timings = std::get<1>(kernel.second);
			auto& result = results[name];
			for (const auto& timing : timings)
			{
				const auto& begin = std::get<0>(timing);
				const auto& end = std::get<1>(timing);
				std::chrono::time_point<std::chrono::high_resolution_clock> zero {};
				if (end == zero) continue;
				auto duration = std::chrono::duration<double, std::micro>{end - begin}.count();
				if (duration <= 0.0) continue;
				assert(duration > 0.0);
				const auto& gridDim = std::get<2>(timing);
				const auto& blockDim = std::get<3>(timing);
				
				result.first = nregisters;
				auto& stats = result.second[std::make_pair(gridDim, blockDim)];
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

	// Conclude and report the results.
	for (auto& result : results)
	{
		const auto& name = result.first;			
		const auto nregisters = result.second.first;

		// TODO If the name is long, shorten it to the last part after ::
		std::cout << name << " (" << nregisters << " registers)" << std::endl;
		
		for (auto& grid : result.second.second)
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

