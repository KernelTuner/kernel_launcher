#include <exception>
#include <fstream>
#include <iterator>
#include <json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include<iostream>
#include <sstream>

#define NVRTC_GET_TYPE_NAME 1
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

namespace kernel_launcher {

std::string curesult_as_string(CUresult code) {
    // cuGetError{Name,String} can fail in which case they do not set the
    // output string to anything. Just set the error to ??? in this case.
    const char *name = "???";
    const char *description = "???";

    cuGetErrorName(code, &name);
    cuGetErrorString(code, &description);

    std::string result;
    result.append(name);
    result.append(": ");
    result.append(description);
    return result;
}

class cuda_exception: public std::exception {
    public:
        cuda_exception(CUresult code): 
            _code(code), 
            _message(curesult_as_string(code)) {

        }

        cuda_exception(CUresult code, const char *message): 
            _code(code), 
            _message(message) {

        }

        const char* what() const noexcept {
            return _message.c_str();
        }

        CUresult code() const noexcept {
            return _code;
        }

    private:
        CUresult _code;
        std::string _message;
};

void cu_check(CUresult code) {
    if (code != CUDA_SUCCESS) {
        throw cuda_exception(code);
    }
}

class nvrtc_exception: public std::exception {
    public:
        nvrtc_exception(nvrtcResult code): 
            _code(code), 
            _message(nvrtcGetErrorString(_code)) {

        }

        nvrtc_exception(nvrtcResult code, const char *message): 
            _code(code), 
            _message(message) {

        }

        const char* what() const noexcept {
            return _message.c_str();
        }

        nvrtcResult code() const noexcept {
            return _code;
        }

    private:
        nvrtcResult _code;
        std::string _message;
};

void nvrtc_check(nvrtcResult code) {
    if (code != NVRTC_SUCCESS) {
        throw nvrtc_exception(code);
    }
}


class CudaFunction {
    public:
        CudaFunction() = default;
        CudaFunction(CudaFunction&& that) {
            std::swap(this->_module, that._module);
            std::swap(this->_module, that._module);
        }

        CudaFunction& operator=(CudaFunction&& that) noexcept {
            std::swap(this->_module, that._module);
            std::swap(this->_module, that._module);
            return *this;
        }

        CudaFunction(const CudaFunction&) = delete;
        CudaFunction& operator=(CudaFunction& that) = delete;


        CudaFunction(const void *ptx_image, const char *function_name) {
            cu_check(cuModuleLoadData(&_module, ptx_image));

            try {
                cu_check(cuModuleGetFunction(&_function, _module, function_name));
            } catch (std::exception &e) {
                cuModuleUnload(_module); // ignore any errors
                throw;
            }

        }

        CUfunction get() const {
            return _function;
        }

        ~CudaFunction() {
            cuModuleUnload(_module); // ignore any errors
        }

    private:
        CUmodule _module = nullptr;
        CUfunction _function = nullptr;

};

class CudaCompiler {
    public:
        CudaCompiler(std::string function_name, std::string file_name, std::string source):
            function_name(function_name),
            file_name(file_name),
            source(source) {
            //
        }

        void add_option(std::string opt) {
            options.push_back(opt);
        }

        void add_options(std::vector<std::string> opts) {
            for (std::string opt: opts) {
                options.push_back(opt);
            }
        }

        void add_header(std::string name, std::string content) {
            header_names.push_back(name);
            header_contents.push_back(content);
        }


        CudaFunction compile() {
            source.push_back('\0');

            std::vector<const char*> hnames;
            std::vector<const char*> hcontents;
            for (size_t i = 0; i < header_names.size(); i++) {
                hnames.push_back(header_names[i].c_str());
                hcontents.push_back(header_contents[i].c_str());
            }

            std::vector<const char*> opts;
            for (auto opt: options) {
                opts.push_back(opt.c_str());
            }

            nvrtcProgram program = nullptr;
            nvrtc_check(nvrtcCreateProgram(
                        &program,
                        source.c_str(),
                        file_name.c_str(),
                        (int) hnames.size(),
                        hnames.data(),
                        hcontents.data()
            ));

            std::vector<char> ptx;
            std::string lowered_name;

            try {
                nvrtc_check(nvrtcAddNameExpression(program, function_name.c_str()));

                nvrtcResult result = nvrtcCompileProgram(program, opts.size(), opts.data());
                if (result != NVRTC_SUCCESS) {
                    size_t log_size = 0;
                    nvrtc_check(nvrtcGetProgramLogSize(program, &log_size));

                    std::vector<char> log(log_size);
                    nvrtc_check(nvrtcGetProgramLog(program, log.data()));

                    log.push_back('\0');
                    throw nvrtc_exception(result, log.data());
                }

                size_t size = 0;
                nvrtc_check(nvrtcGetPTXSize(program, &size));

                ptx.resize(size);
                nvrtc_check(nvrtcGetPTX(program, ptx.data()));


                // We must copy name to a new string allocation since 
                // nvrtcDestroyProgram will free the pointer returned
                // by nvrtcGetLoweredName.
                const char *name = nullptr;
                nvrtc_check(nvrtcGetLoweredName(program, function_name.c_str(), &name));
                lowered_name = name;

                nvrtc_check(nvrtcDestroyProgram(&program));

            } catch (std::exception &e) {
                nvrtcDestroyProgram(&program);
                throw;
            }

            return CudaFunction(ptx.data(), lowered_name.c_str());
        }

    private:
        std::string function_name;
        std::string file_name;
        std::string source;
        std::vector<std::string> header_names;
        std::vector<std::string> header_contents;
        std::vector<std::string> options;

};


CudaFunction compile_kernel(
    const std::string kernel_name, 
    const std::string kernel_file, 
    const std::string problem_size,
    const std::string results_file,
    const std::unordered_map<std::string, std::string> params,
    const std::vector<std::string> compiler_flags
) {
    // open file
    std::ifstream ifs(kernel_file);
    if (!ifs) {
        throw std::runtime_error("error while reading: " + kernel_file);
    }

    // read source code.
    std::vector<char> source((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    source.push_back('\0');

    CudaCompiler compiler(kernel_name, kernel_file, source.data());

    // read results file
    // parse results
    // TODO: actually do something with results_file
    
    // Define parameters as marcos.
    for (auto &pair: params) {
        compiler.add_option("--define-macro=" + pair.first + "=" + pair.second);
    }


    // Add sm_<major><minor> of the current device as compile flag.
    CUdevice device;
    int major, minor;
    cu_check(cuCtxGetDevice(&device));
    cu_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    cu_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));

    std::stringstream arch;
    arch << "--gpu-architecture=compute_" << major << minor;
    compiler.add_option(arch.str());

    // add remaining compiler flags
    for (const std::string &opt: compiler_flags) {
        compiler.add_option(opt);
    }

    return compiler.compile();
}



class RawKernel {
    public:
        RawKernel() = default;

        RawKernel(
                const std::string kernel_name, 
                const std::string kernel_file, 
                const std::string problem_size,
                const std::string results_file = "",
                const std::unordered_map<std::string, std::string> params = {},
                const std::vector<std::string> compiler_flags = {}
        ): kernel(compile_kernel(
            kernel_name,
            kernel_file,
            problem_size,
            results_file,
            params,
            compiler_flags
        )) {
            //
        }

        void launch(dim3 grid, dim3 block, void **args) {
            launch_with_shared_memory(grid, block, 0, args);
        }

        void launch_async(dim3 grid, dim3 block, CUstream stream, void **args) {
            launch_with_shared_memory_async(grid, block, 0, stream, args);
        }

        void launch_with_shared_memory(dim3 grid, dim3 block, unsigned int shared_mem, void **args) {
            launch_with_shared_memory_async(grid, block, shared_mem, nullptr, args);
            cu_check(cuStreamSynchronize(nullptr));
        }

        void launch_with_shared_memory_async(dim3 grid, dim3 block, unsigned int shared_mem, CUstream stream, void **args) {
            cu_check(cuLaunchKernel(
                        kernel.get(),
                        grid.x, 
                        grid.y, 
                        grid.z,
                        block.x, 
                        block.y, 
                        block.z,
                        shared_mem,
                        stream,
                        args,
                        nullptr
            ));
        }

    private:
        CudaFunction kernel;
};


template <typename ...Args>
struct collect_types_impl;

template <typename T, typename ...Rest>
struct collect_types_impl<T, Rest...> {
    static void call(std::vector<std::string> &output) {
        std::string result;
        nvrtc_check(nvrtcGetTypeName<T>(&result));
        output.push_back(result);

        collect_types_impl<Rest...>::call(output);
    }
};

template <>
struct collect_types_impl<> {
    static void call(std::vector<std::string> &output) {
        (void) output;
    }
};

template <typename ...Args>
std::string generate_typed_kernel_name(std::string function_name) {
    std::string name = "(void(*)(";
    std::vector<std::string> types;
    collect_types_impl<Args...>::call(types);

    for (size_t i = 0; i < types.size(); i++) {
        if (i != 0) {
            name.append(",");
        }

        name.append(types[i]);
    }

    name.append("))");
    name.append(function_name);
    return name;
}

template <typename ...Args>
struct pack_args_impl;

template <typename T, typename ...Rest>
struct pack_args_impl<T, Rest...> {
    static void call(std::vector<void*> &output, T &arg, Rest&... rest) {
        output.push_back((void*)(T*) &arg);
        pack_args_impl<Rest...>::call(output, rest...);
    }
};

template <>
struct pack_args_impl<> {
    static void call(std::vector<void*> &output) {
        (void) output;
    }
};

template <typename ...Args>
std::vector<void*> pack_args(Args&... args) {
    std::vector<void*> output;
    pack_args_impl<Args...>::call(output, args...);
    return output;
}

template <typename ...Args>
class Kernel {
    public:
        Kernel() = default;

        Kernel(
                const std::string kernel_name, 
                const std::string kernel_file, 
                const std::string problem_size,
                const std::string results_file = "",
                const std::unordered_map<std::string, std::string> params = {},
                const std::vector<std::string> compiler_flags = {}
        ): kernel(
            generate_typed_kernel_name<Args...>(kernel_name),
            kernel_file,
            problem_size,
            results_file,
            params,
            compiler_flags
        ) {
            //
        }

        void launch(dim3 grid, dim3 block, Args&... args) {
            std::vector<void*> ptrs = pack_args(args...);
            kernel.launch(grid, block, ptrs.data());
        }

        void launch_async(dim3 grid, dim3 block, CUstream stream, Args&... args) {
            std::vector<void*> ptrs = pack_args(args...);
            kernel.launch_async(grid, block, stream, ptrs.data());
        }

        void launch_with_shared_memory(dim3 grid, dim3 block, unsigned int shared_mem, Args&... args) {
            std::vector<void*> ptrs = pack_args(args...);
            kernel.launch_with_shared_memory(grid, block, shared_mem, ptrs.data());
        }

        void launch_with_shared_memory_async(dim3 grid, dim3 block, unsigned int shared_mem, CUstream stream, Args&... args) {
            std::vector<void*> ptrs = pack_args(args...);
            kernel.launch_with_shared_memory_async(grid, block, shared_mem, stream, ptrs.data());
        }


    private:
        RawKernel kernel;
};




}
