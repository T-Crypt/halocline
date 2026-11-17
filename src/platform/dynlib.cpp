// src/platform/dynlib.cpp - see include/halo/platform/dynlib.hpp.
#include "halo/platform/dynlib.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "halo/platform/file.hpp"

#include <dlfcn.h>
#include <link.h>

namespace halo::platform {

DynLib DynLib::open_nvml() {
    DynLib lib;
    lib.h_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    return lib;
}

DynLib DynLib::open_app(const std::string& file) {
    DynLib lib;
    lib.h_ = dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL);
    return lib;
}

std::string DynLib::versioned_name(const char* name, int major) { return "lib" + std::string(name) + ".so." + std::to_string(major); }

std::string DynLib::path() const {
    struct link_map* lm = nullptr;
    if (h_ == nullptr || dlinfo(h_, RTLD_DI_LINKMAP, &lm) != 0 || lm == nullptr || lm->l_name == nullptr) return std::string();
    return lm->l_name;
}

void* DynLib::address(const char* name) const { return h_ == nullptr ? nullptr : dlsym(h_, name); }

void DynLib::reset() { h_ = nullptr; }   // never dlclosed, as before: the management library stays loaded

}  // namespace halo::platform
