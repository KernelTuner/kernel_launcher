def rst_comment():
    text = "This file has been auto-generated. DO NOT MODIFY ITS CONTENT"
    bars = "=" * len(text)
    return f"..\n  {bars}\n  {text}\n  {bars}\n\n"


def build_doxygen_page(name, items):
    content = rst_comment()
    content += f".. _{name}:\n\n"
    content += name + "\n" + "=" * len(name) + "\n"

    for item in items:
        directive = "doxygenstruct" if item[0].isupper() else "doxygenfunction"
        content += f".. {directive}:: kernel_launcher::{item}\n"

    filename = f"api/{name}.rst"
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
            if isinstance(symbol, str):
                name = symbol
                items = [symbol]
            else:
                name, items = symbol

            filename = build_doxygen_page(name, items)
            children.append(filename)

            filename = filename.replace(".rst", "")
            body += f"* :doc:`{name} <{filename}>`\n"

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
    "Wisdom Kernels": [
        "WisdomKernel",
        "WisdomSettings",
        "WisdomRecord",
        "Oracle",
        "DefaultOracle",
        "load_best_config",
        "process_wisdom_file",
        "default_wisdom_settings",
        "append_global_wisdom_directory",
        "set_global_capture_directory",
        "add_global_capture_pattern",
        "export_capture_file",
        "capture_file_exists",
    ],
    "Pragma Kernels": [
        "PragmaKernel",
        "build_pragma_kernel"
    ],
    "Registry": [
        "KernelRegistry",
        "IKernelDescriptor",
        "default_registry",
        "launch",
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
        ("cuda_copy", ["cuda_copy(CudaSpan<T>, CudaSpan<const T>)", "cuda_copy(const T*, T*, size_t)"]),
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
        ("type_of", ["type_of()", "type_of(const T&)"]),
        ("type_name", ["type_name()", "type_name(const T&)"]),
        "Value",
        "Variable",
        "KernelArg",
        "IntoKernelArg",
        "into_kernel_arg",
    ]
}


build_index_page(groups)
