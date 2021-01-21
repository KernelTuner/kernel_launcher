#include <cuda.h>
#include <exception>
#include <fstream>
#include <iterator>
#include <json.hpp>
#include <nvrtc.h>
#include <string>
#include <unordered_map>
#include <vector>
#include<iostream>

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

class cu_exception: public std::exception {
    public:
        cu_exception(CUresult code): 
            _code(code), 
            _message(curesult_as_string(code)) {

        }

        cu_exception(CUresult code, const char *message): 
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
        throw cu_exception(code);
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


struct compilation_result_t {
    std::vector<char> ptx;
    std::unordered_map<std::string, std::string> expressions;
};


compilation_result_t compile_kernel(
    const char *path,
    const std::vector<const char*> expressions,
    const std::vector<const char*> options
) {
    int num_headers = 0;
    const char **headers = nullptr; 
    const char **include_names = nullptr;

    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("error while reading " + std::string(path));
    }

    std::vector<char> src(
            std::istreambuf_iterator<char>(ifs),
            (std::istreambuf_iterator<char>()));
    src.push_back('\0');

    nvrtcProgram program = nullptr;
    nvrtc_check(nvrtcCreateProgram(
                &program,
                src.data(),
                path,
                num_headers,
                headers,
                include_names));

    try {
        for (const std::string &expr: expressions) {
            nvrtc_check(nvrtcAddNameExpression(program, expr.c_str()));
        }

        nvrtcResult result = nvrtcCompileProgram(program, options.size(), options.data());
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

        std::vector<char> ptx(size);
        nvrtc_check(nvrtcGetPTX(program, ptx.data()));

        std::unordered_map<std::string, std::string> mapping;
        for (const std::string &expr: expressions) {
            const char *lowered_name = nullptr;
            nvrtc_check(nvrtcGetLoweredName(program, expr.c_str(), &lowered_name));

            mapping[expr] = std::string(lowered_name);
        }

        nvrtc_check(nvrtcDestroyProgram(&program));

        return compilation_result_t {
            ptx,
            mapping
        };
    } catch (std::exception &e) {
        nvrtcDestroyProgram(&program);
        throw;
    }
}

class KernelLauncher {
    public:
        KernelLauncher(std::string path): KernelLauncher(path.c_str()) {
            //
        }

        KernelLauncher(const char *path) {
            compile_kernel(
                    path,
                    {},
                    {});
        }
};

}
