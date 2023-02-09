def rst_comment():
    text = "This file has been auto-generated. DO NOT MODIFY ITS CONTENT"
    bars = "=" * len(text)
    return f"..\n  {bars}\n  {text}\n  {bars}\n\n"

def build_doxygen_page(symbol, directive):
    content = rst_comment()
    content += f".. _{symbol}:\n\n"
    content += symbol + "\n" + "=" * len(symbol) + "\n"
    content += f".. {directive}:: kernel_launcher::{symbol}\n"

    filename = f"api/{symbol}.rst"
    print(f"writing to {filename}")

    with open(filename, "w") as f:
        f.write(content)

    return filename


def build_index_page(groups):
    body = ""
    children = []

    for groupname, symbols in groups.items():
        body += f".. raw:: html\n\n   <h2>{groupname}</h2>\n\n"

        for symbol in symbols:
            directive = "doxygenstruct" if symbol[0].isupper() else "doxygenfunction"
            filename = build_doxygen_page(symbol, directive)
            children.append(filename)

            filename = filename.replace(".rst", "")
            body += f"* :doc:`{symbol} <{filename}>`\n"

        body += "\n"


    title = "API Reference"
    content = rst_comment()
    content += title + "\n" + "=" * len(title) + "\n"
    content += ".. toctree::\n"
    content += "   :titlesonly:\n"
    content += "   :hidden:\n\n"

    for filename in sorted(children):
        content += f"   {filename}\n"

    content += "\n"
    content += body + "\n"

    filename = "api.rst"
    print(f"writing to {filename}")

    with open(filename, "w") as f:
        f.write(content)

    return filename


groups = {
    "Kernel" : [
        "ConfigSpace",
        "Config",
        "KernelBuilder",
        #"KernelDef",
        "KernelInstance",
        "KernelSource",
        "Kernel",
    ],
    "Wisdom": [
        "WisdomKernel",
        "WisdomSettings",
        "WisdomRecord",
        "Oracle",
        "DefaultOracle",
        "append_global_wisdom_directory",
        "default_wisdom_settings",
        "export_tuning_file",
        "load_best_config",
        "process_wisdom_file",
        "set_global_wisdom_directory",
        "tuning_file_exists",
        "set_global_capture_directory",
        "add_global_capture_pattern",
    ],
    "Registry": [
        "default_registry",
        "KernelRegistry",
        "IKernelDescriptor",
        #"KernelDescriptor",
    ],
    "Compilation": [
        "ICompiler",
        "Compiler",
        "NvrtcCompiler",
        "NvrtcException",
        "default_compiler",
        "set_global_default_compiler",
    ],
    "CUDA Utilities": [
        "CudaArch",
        "CudaContextHandle",
        "CudaDevice",
        "CudaException",
        "CudaSpan",
        "cuda_check",
        "cuda_copy",
        "cuda_span",
    ],
    "Utilities": [
        "FileLoader",
        "DefaultLoader",
        #"ForwardLoader",
        "ProblemSize",
        "TemplateArg",
        "TunableParam",
        "TypeInfo",
        "Value",
        "Variable",
        "KernelArg",
        "IntoKernelArg",
        "into_kernel_arg",
    ]
}


build_index_page(groups)
