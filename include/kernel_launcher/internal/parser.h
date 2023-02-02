#ifndef KERNEL_LAUNCHER_PARSER_H
#define KERNEL_LAUNCHER_PARSER_H

#include "../value.h"
#include "tokens.h"

namespace kernel_launcher {
namespace internal {

struct TemplateParam {
    Token name;
    bool is_integral;
    std::string integral_type;
    Value default_value;
};

struct FunctionParam {
    std::string type;
    Token name;
};

struct KernelDef {
    std::string qualified_name;
    Token name;
    std::vector<Token> directives;
    std::vector<TemplateParam> template_params;
    std::vector<FunctionParam> fun_params;
};

std::vector<KernelDef> parse_kernels(TokenStream& stream);

}  // namespace internal
}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_PARSER_H
