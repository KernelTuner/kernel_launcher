#ifndef INCLUDE_KERNEL_LAUNCHER_H_
#define INCLUDE_KERNEL_LAUNCHER_H_

//#define NVRTC_GET_TYPE_NAME 1
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <json.hpp>

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
        explicit cuda_exception(CUresult code):
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
        explicit nvrtc_exception(nvrtcResult code):
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
        Config() = default;
        explicit Config(std::unordered_map<std::string, int64_t> params):
                params(params) {
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

        static Config select_best(
                const json &props,
                const std::string &problem_size,
                std::string device_name
        ) {
            static const int INVALID = -1;
            static const int VALID = 1;
            static const int VALID_DEVICE = 2;
            static const int VALID_DEVICE_AND_PROBLEM = 3;

            for (char &c: device_name) {
                if (c == ' ' || c == '-') {
                    c = '_';
                }
            }

            if (props.at("version_number") != "1.0") {
                throw std::runtime_error("JSON file has invalid format, expecting version 1.0");
            }

            const std::string &objective = props.at("objective");
            bool higher_is_better = props.at("objective_higher_is_better");
            std::vector<std::string> param_keys = props.at("tunable_parameters");
            const json &records = props.at("data");  // Tuning results

            const json *best_record = nullptr;
            int best_type = INVALID;
            double best_score = 0.0;

            for (size_t i = 0; i < records.size(); i++) {
                const json &record = records.at(i);
                int type = VALID;

                if (record.at("device_name") == device_name) {
                    type = VALID_DEVICE;

                    if (record.at("problem_size") == problem_size) {
                        type = VALID_DEVICE_AND_PROBLEM;
                    }
                }

                double score = record.at(objective);

                if (type > best_type ||
                        (higher_is_better && score > best_score) ||
                        (!higher_is_better && score < best_score)) {
                    best_score = score;
                    best_type = type;
                    best_record = &record;
                }
            }

            if (best_record == nullptr) {
                throw std::runtime_error("could not load configuration, not valid results found");
            } else if (best_type == VALID) {
                std::cerr << "WARNING: GPU " << device_name << " not found in "
                    << "tuning results, selecting best configuration across "
                    << "all GPUs." << std::endl;
            } else if (best_type == VALID_DEVICE) {
                std::cerr << "WARNING: problem " << problem_size << " not found in "
                    << "tuning results, selecting best configuration across all problem sizes "
                    << "for GPU " << device_name
                    << std::endl;
            }


            const json &best_config = best_record->at("tunable_parameters");
            Config params;

            for (const std::string &key: param_keys) {
                int64_t value = best_config.at(key);
                params.set(key, value);
            }

            params.kernel_name = props.at("kernel_name");
            return params;
        }

        static Config load_best(
                const std::string &file_name,
                const std::string &problem_size,
                const std::string &device_name
        ) {
            std::ifstream ifs(file_name);
            if (!ifs) {
                throw std::runtime_error("error while reading: " + file_name);
            }

            json results = json::parse(ifs);

            return Config::select_best(results, problem_size, device_name);
        }

        static Config load_best_for_current_device(
                const std::string &file_name,
                const std::string &problem_size
        ) {
            CUdevice device;
            cu_check(cuCtxGetDevice(&device));

            static char name[256];
            cu_check(cuDeviceGetName(name, sizeof(name), device));

            return Config::load_best(file_name, problem_size, name);
        }

    //private:
        std::unordered_map<std::string, int64_t> params;
        std::string kernel_name;
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
        CudaModule& operator=(const CudaModule&) = delete;


        CudaModule(const void *ptx_image, const char *function_name) {
            cu_check(cuModuleLoadData(&_module, ptx_image));

            try {
                cu_check(cuModuleGetFunction(&_function, _module, function_name));
            } catch (std::exception &e) {
                cuModuleUnload(_module);  // ignore any errors
                throw;
            }
        }

        CUfunction get() const {
            return _function;
        }

        ~CudaModule() {
            cuModuleUnload(_module);  // ignore any errors
        }

    private:
        CUmodule _module = nullptr;
        CUfunction _function = nullptr;
};


class CudaCompiler {
    public:
        CudaCompiler(std::string function_name, std::string kernel_file):
            function_name(function_name),
            file_name(kernel_file) {
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
            std::ifstream ifs(file_name);
            if (!ifs) {
                throw std::runtime_error("error while reading: " + file_name);
            }

            std::vector<char> source((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
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
                        source.data(),
                        file_name.c_str(),
                        static_cast<int>(hnames.size()),
                        hnames.data(),
                        hcontents.data()));

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
        std::vector<std::string> header_names;
        std::vector<std::string> header_contents;
        std::vector<std::string> options;
};


class RawKernel {
    public:
        static RawKernel compile(
            const Config &&params,
            const std::string &kernel_name,
            const std::string &kernel_file,
            const std::vector<std::string> &compiler_flags = {}
        ) {
            // Create compiler instance
            CudaCompiler compiler(kernel_name, kernel_file);

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

            CudaModule module = compiler.compile();
            return RawKernel(std::move(module), std::move(params));
        }

        RawKernel() = default;

        RawKernel(CudaModule kernel, Config config):
                kernel(std::move(kernel)), 
                config(std::move(config)) {
            //
        }

        CUresult launch_unchecked(dim3 grid, unsigned int shared_mem, CUstream stream, void **args) const {
            dim3 block = config.get_block_dim();

            return cuLaunchKernel(
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
                        nullptr);
        }

        const Config& get_config() const {
            return config;
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
        output.push_back(reinterpret_cast<void*>(&arg));
        pack_args_impl<Rest...>::call(output, rest...);
    }
};

template <>
struct pack_args_impl<> {
    static void call(std::vector<void*> &output) {
        (void) output;
    }
};
}  // namespace detail


template <typename ...Args>
class KernelLaunch {
    public:
        KernelLaunch(dim3 grid, unsigned int shared_mem, CUstream stream, bool synchronize, const RawKernel *kernel):
            grid(grid),
            shared_mem(shared_mem),
            stream(stream),
            synchronize(synchronize),
            kernel(kernel) {
            //
        }

        CUresult launch_unchecked(Args... args) const {
            std::vector<void*> ptrs;
            detail::pack_args_impl<Args...>::call(ptrs, args...);

            return kernel->launch_unchecked(grid, shared_mem, stream, ptrs.data());
        }

        void launch(Args... args) const {
            cu_check(launch_unchecked(args...));

            if (synchronize) {
                cu_check(cuStreamSynchronize(stream));
            }
        }

        void operator()(Args... args) const {
            launch(args...);
        }

    private:
        const dim3 grid;
        const unsigned int shared_mem;
        const CUstream stream;
        const bool synchronize;
        const RawKernel *kernel;
};


template <typename ...Args>
class Kernel {
    public:
        static Kernel compile(
                const Config &&config,
                const std::string &kernel_file,
                const std::vector<std::string> &compiler_flags = {}) {
            std::string kernel_name = detail::generate_typed_kernel_name<Args...>(config.kernel_name);

            return Kernel(RawKernel::compile(
                    std::move(config),
                    kernel_name,
                    kernel_file,
                    compiler_flags
            ));
        }

        static Kernel compile_best_for_current_device(
                const std::string &tuning_file,
                const std::string &problem_size,
                const std::string &kernel_file,
                const std::vector<std::string> &compiler_flags = {}) {
            auto config = Config::load_best_for_current_device(tuning_file, problem_size);
            return Kernel::compile(std::move(config), kernel_file, compiler_flags);
        }


        Kernel() = default;

        explicit Kernel(RawKernel inner): kernel(std::move(inner)) {
            //
        }

        KernelLaunch<Args...> configure(dim3 grid) const {
            return configure(grid, 0);
        }

        KernelLaunch<Args...> configure(dim3 grid, unsigned int shared_mem) const {
            return KernelLaunch<Args...>(grid, shared_mem, 0, true, &kernel);
        }

        KernelLaunch<Args...> configure_async(dim3 grid, CUstream stream) const {
            return configure_async(grid, 0, stream);
        }

        KernelLaunch<Args...> configure_async(dim3 grid, unsigned int shared_mem, CUstream stream) const {
            return KernelLaunch<Args...>(grid, shared_mem, stream, false, &kernel);
        }

        KernelLaunch<Args...> operator()(dim3 grid) const {
            return configure(grid);
        }

        KernelLaunch<Args...> operator()(dim3 grid, CUstream stream) const {
            return configure_async(grid, 0, stream);
        }

        KernelLaunch<Args...> operator()(dim3 grid, unsigned int shared_mem) const {
            return configure(grid, shared_mem);
        }

        KernelLaunch<Args...> operator()(dim3 grid, unsigned int shared_mem, CUstream stream) const {
            return configure_async(grid, shared_mem, stream);
        }

        dim3 get_block_dim() const {
            return get_config().get_block_dim();
        }

        const Config& get_config() const {
            return kernel.get_config();
        }

    private:
        RawKernel kernel;
};



}  // namespace kernel_launcher

#endif  // INCLUDE_KERNEL_LAUNCHER_H_
