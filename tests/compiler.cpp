#include "kernel_launcher/compiler.h"

#include "catch.hpp"

using namespace kernel_launcher;

TEST_CASE("test NvrtcCompiler") {
    NvrtcCompiler c = NvrtcCompiler {};

    KernelSource kernel_source {"kernels/test.cu"};
    std::string kernel_name = "dummy";
    std::vector<TemplateArg> template_args;
    std::vector<TypeInfo> param_types;
    std::vector<std::string> options;
    std::string ptx;
    std::string symbol_name;

    c.compile_ptx(
        KernelDef {
            kernel_name,
            kernel_source,
            template_args,
            param_types,
            options,
        },
        ptx,
        symbol_name);
}