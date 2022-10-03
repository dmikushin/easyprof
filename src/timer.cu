#include "easyprof.h"

#include <algorithm>
#include <iostream>

static bool operator<(const dim3& v1, const dim3& v2)
{
	auto c1 = reinterpret_cast<const char*>(&v1);
	auto c2 = reinterpret_cast<const char*>(&v2);
	return memcmp(c1, c2, sizeof(dim3)) > 0;
}

bool Timer::isTiming() { return timing; }

Timer::Timer(const std::map<void*, std::shared_ptr<GPUfunction>>& funcs_) : funcs(funcs_)
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

void Timer::measure(const GPUfunction* func_,
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

void Timer::sync(gpuStream_t stream)
{
	if (synced) return;

	auto end = std::chrono::high_resolution_clock::now();
	
	// Synchonize all kernels of stream, which are not yet synchronized.
	for (auto& pair : kernels[stream])
	{
		// const auto& name = pair.first;
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

void Timer::sync()
{
	// Sync all outstanding streams.
	for (auto& pair : kernels)
		sync(pair.first);
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
			auto& result = results[name];
			for (const auto& timing : timings)
			{
				const auto& begin = std::get<0>(timing);
				const auto& end = std::get<1>(timing);
				auto duration = std::chrono::duration<double, std::micro>{end - begin}.count();
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

