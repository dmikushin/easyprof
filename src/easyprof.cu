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

Profiler profiler;

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

std::map<std::string, void*> dlls;

const char* __dll(const char* prefix)
{
	std::string sprefix = prefix;
	if (sprefix == "cu") return LIBGPU;
	if (sprefix == "cuda") return LIBGPURT;
	if (sprefix == "hip") return LIBGPURT;
	return "";
}


