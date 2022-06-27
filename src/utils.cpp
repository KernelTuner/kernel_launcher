#include <stdexcept>

#if __clang__ || __GNUC__
    #include <cxxabi.h>
#endif

#include "kernel_launcher/utils.h"

namespace kernel_launcher {

std::string demangle_type_info(const std::type_info& type) {
#if __clang__ || __GNUC__
    const char* mangled_name = type.name();
    int status = ~0;
    // TOOD: look into how portable this solution is on different platforms :-/
    char* undecorated_name =
        abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

    // status == 0: OK
    // status == -1: memory allocation failure
    // status == -2: name is invalid
    // status == -3: one of the other arguments is invalid
    if (status != 0) {
        throw std::runtime_error(
            std::string("__cxa_demangle failed for ") + mangled_name);
    }

    std::string result = undecorated_name;
    free(undecorated_name);
    return result;
#else
    throw std::runtime_error("this platform does not support `__cxa_demangle`");
#endif
}

}  // namespace kernel_launcher