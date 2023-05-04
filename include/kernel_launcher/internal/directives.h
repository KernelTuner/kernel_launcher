#ifndef KERNEL_LAUNCHER_DIRECTIVES_H
#define KERNEL_LAUNCHER_DIRECTIVES_H

#include "../builder.h"
#include "parser.h"
#include "tokens.h"

namespace kernel_launcher {
namespace internal {

KernelBuilder builder_from_annotated_kernel(
    TokenStream& stream,
    KernelSource source,
    const AnnotatedKernelSpec& def,
    const std::vector<Value>& template_args);

}
}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_DIRECTIVES_H
