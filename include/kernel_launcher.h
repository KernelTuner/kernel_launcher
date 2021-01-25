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
using json = nlohmann::json;

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

        cuda_exception(CUresult code, const std::string &message): 
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

        nvrtc_exception(nvrtcResult code, const std::string &message): 
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



class Config {

    public:
        static Config load_best(
                const json &results, 
                const std::string &problem_size, 
                const std::string &objective,
                std::string device_name
        ) {
            for (char &c: device_name) {
                if (c == ' ' || c == '-') {
                    c = '_';
                }
            }

            std::vector<json> options;

            if (!results.contains(device_name)) {
                std::cerr << "WARNING: GPU " << device_name << "not found in " 
                    << "tuning results, selecting best configuration across all GPUs."
                    << std::endl;

                for (auto &x: results) {
                    for (auto &y: x) {
                        for (auto &z: y) {
                            options.push_back(z);
                        }
                    }
                }
            } else if (!results[device_name].contains(problem_size)) {
                std::cerr << "WARNING: problem " << problem_size << "not found in " 
                    << "tuning results, selecting best configuration across all GPUs."
                    << std::endl;

                for (auto &x: results[device_name]) {
                    for (auto &y: x) {
                        options.push_back(y);
                    }
                }

            } else {
                for (auto &x: results[device_name][problem_size]) {
                    options.push_back(x);
                }
            }

            json *best_config = nullptr;
            double best_score = 0;

            for (auto &option: options) {
                double score = option[objective];

                if (score > best_score) {
                    best_config = &option;
                    best_score = score;
                }
            }

            std::unordered_map<std::string, int64_t> params;
            for (auto &it: best_config->items()) {
                try {
                    auto key = it.key();

                    if (key != objective) {
                        int64_t value = 0;
                        it.value().get_to(value);
                        params[key] = value;
                    }
                } catch (const json::exception &e) {
                    // ignore any type conversion errors.
                }
            }

            return Config(params);
        }

        static Config load_best(
                const std::string &file_name, 
                const std::string &problem_size, 
                const std::string &objective, 
                const std::string &device_name
        ) {
            std::ifstream ifs(file_name);
            if (!ifs) {
                throw std::runtime_error("error while reading: " + file_name);
            }

            json results = json::parse(ifs);

            return Config::load_best(results, problem_size, objective, device_name);
        }

        static Config load_best_for_current_device(
                const std::string &file_name, 
                const std::string &problem_size, 
                const std::string &objective
        ) {
            CUdevice device;
            cu_check(cuCtxGetDevice(&device));

            static char name[256];
            cu_check(cuDeviceGetName(name, sizeof(name), device));

            return Config::load_best(file_name, problem_size, objective, name);
        }


        Config() {
            //
        }

        Config(std::unordered_map<std::string, int64_t> params): params(params) {
            //
        }

        void set(std::string key, int64_t value) {
            params[key] = value;
        }

        int64_t get(const std::string &key) const {
            return params.at(key);
        }

        int64_t get(const std::string &key, int64_t def) const {
            auto it = params.find(key);

            if (it == params.end()) {
                return def;
            } else {
                return it->second;
            }
        }


        dim3 get_block_dim() const {
            dim3 block;
            block.x = get("block_size_x", 1);
            block.y = get("block_size_y", 1);
            block.z = get("block_size_z", 1);
            return block;
        }

        const std::unordered_map<std::string, int64_t>& get_all() const {
            return params;
        }


    private:
        std::unordered_map<std::string, int64_t> params;
};


class CudaModule {
    public:
        CudaModule() = default;
        CudaModule(CudaModule&& that) {
            std::swap(this->_module, that._module);
            std::swap(this->_function, that._function);
        }

        CudaModule& operator=(CudaModule&& that) noexcept {
            std::swap(this->_module, that._module);
            std::swap(this->_function, that._function);
            return *this;
        }

        CudaModule(const CudaModule&) = delete;
        CudaModule& operator=(CudaModule& that) = delete;


        CudaModule(const void *ptx_image, const char *function_name) {
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

        ~CudaModule() {
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


        CudaModule compile() {
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

            return CudaModule(ptx.data(), lowered_name.c_str());
        }

    private:
        std::string function_name;
        std::string file_name;
        std::string source;
        std::vector<std::string> header_names;
        std::vector<std::string> header_contents;
        std::vector<std::string> options;

};


CudaModule compile(
    const std::string &kernel_name, 
    const std::string &kernel_file, 
    const Config &params,
    const std::vector<std::string> &compiler_flags = {}
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
    for (auto &pair: params.get_all()) {
        std::stringstream option;
        option << "--define-macro=" << pair.first << "=" << pair.second;
        compiler.add_option(option.str());
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
                const std::string &kernel_name, 
                const std::string &kernel_file, 
                const Config params,
                const std::vector<std::string> compiler_flags = {}
        ): kernel(compile(
            kernel_name,
            kernel_file,
            params,
            compiler_flags
        )), config(params) {
            //
        }

        void launch(dim3 grid, unsigned int shared_mem, CUstream stream, void **args) const {
            dim3 block = config.get_block_dim();

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
        CudaModule kernel;
        Config config;
};

namespace detail {
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
}

template <typename ...Args>
class Kernel {
    public:
        Kernel() = default;

        Kernel(
                const std::string &kernel_name, 
                const std::string &kernel_file, 
                const Config &params,
                const std::vector<std::string> compiler_flags = {}
        ): kernel(
            detail::generate_typed_kernel_name<Args...>(kernel_name),
            kernel_file,
            params,
            compiler_flags
        ) {
            //
        }

        void launch(dim3 grid, Args... args) const {
            launch_with_shared_memory(grid, 0, args...);
        }

        void launch_with_shared_memory(dim3 grid, unsigned int shared_mem, Args... args) const {
            launch_with_shared_memory_async(grid, shared_mem, 0, args...);
            cu_check(cuStreamSynchronize(nullptr));
        }

        void launch_async(dim3 grid, CUstream stream, Args... args) const {
            launch_with_shared_memory_async(grid, 0, stream, args...);
        }

        void launch_with_shared_memory_async(dim3 grid, unsigned int shared_mem, CUstream stream, Args... args) const {
            std::vector<void*> ptrs;
            detail::pack_args_impl<Args...>::call(ptrs, args...);

            kernel.launch(grid, shared_mem, stream, ptrs.data());
        }

        void operator()(dim3 grid, Args... args) const {
            launch(grid, args...);
        }

        void operator()(dim3 grid, CUstream stream, Args... args) const {
            launch(grid, args...);
        }

        void operator()(dim3 grid, unsigned int shared_mem, Args... args) const {
            launch_with_shared_memory(grid, shared_mem, args...);
        }

        void operator()(dim3 grid, unsigned int shared_mem, CUstream stream, Args... args) const {
            launch_with_shared_memory_async(grid, shared_mem, stream, args...);
        }


    private:
        RawKernel kernel;
};




}
