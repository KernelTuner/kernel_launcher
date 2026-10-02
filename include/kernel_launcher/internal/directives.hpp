#ifndef KERNEL_LAUNCHER_DIRECTIVES_H
#define KERNEL_LAUNCHER_DIRECTIVES_H

#include "kernel_launcher/internal/parser.hpp"
#include "kernel_launcher/internal/tokens.hpp"
#include "kernel_launcher/kernel_builder.hpp"

namespace kernel_launcher {
namespace internal {

KernelBuilder builder_from_annotated_kernel(
    TokenStream& stream,
    FileSource source,
    const AnnotatedKernelSpec& def,
    const std::vector<Primitive>& template_args);

}  // namespace internal
}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_DIRECTIVES_H
