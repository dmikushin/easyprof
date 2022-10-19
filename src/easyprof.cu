#include <map>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <tuple>
#include <vector>

#include "easyprof.h"

void* libdl = nullptr;
void* libgpu = nullptr;
void* libgpurt = nullptr;

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

Profiler& Profiler::get()
{
	static Profiler profiler;
	return profiler;
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


