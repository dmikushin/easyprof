#include "easyprof.h"

#include <regex>

bool Matcher::isMatching(const std::string& name)
{
	if (pattern == "") return true;
	
	std::smatch m;
	const std::regex r(pattern);
	if (std::regex_match(name, m, r))
		return true;
	
	return false;
}

Matcher::Matcher(const std::map<void*, std::shared_ptr<GPUfunction>>& funcs_) : funcs(funcs_)
{
	const char* cpattern = getenv("PROFILE_REGEX");
	if (cpattern) pattern = cpattern;
}

