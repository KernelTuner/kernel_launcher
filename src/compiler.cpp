#include "kernel_launcher/compiler.h"

#include <nvrtc.h>

#include <cstring>
#include <sstream>

namespace kernel_launcher {

inline void nvrtc_check(nvrtcResult result) {
    if (result != NVRTC_SUCCESS) {
        std::stringstream ss;
        ss << "NVRTC error: " << nvrtcGetErrorString(result);
        throw NvrtcException(ss.str());
    }
}

static bool nvrtc_compile(
    const std::string& kernel_source,
    const std::string& kernel_file,
    const std::string& symbol_name,
    const std::vector<const char*>& options,
    const std::vector<const char*>& headers_names,
    const std::vector<const char*>& headers_content,
    std::string& lowered_name,
    std::string& ptx,
    std::string& log) {
    nvrtcProgram program = nullptr;
    nvrtc_check(nvrtcCreateProgram(
        &program,
        kernel_source.c_str(),
        kernel_file.c_str(),
        (int)std::min(headers_names.size(), headers_content.size()),
        headers_names.data(),
        headers_content.data()));

    try {
        nvrtc_check(nvrtcAddNameExpression(program, symbol_name.data()));
        nvrtcResult result =
            nvrtcCompileProgram(program, (int)options.size(), options.data());

        if (result != NVRTC_SUCCESS && result != NVRTC_ERROR_COMPILATION) {
            nvrtc_check(result);
        }

        size_t log_size = 0;
        nvrtc_check(nvrtcGetProgramLogSize(program, &log_size));
        log.resize(log_size + 1);
        nvrtc_check(nvrtcGetProgramLog(program, &log[0]));

        if (result == NVRTC_SUCCESS) {
            const char* ptr = nullptr;
            nvrtc_check(nvrtcGetLoweredName(program, symbol_name.data(), &ptr));
            lowered_name = ptr;

            size_t ptx_size = 0;
            nvrtc_check(nvrtcGetPTXSize(program, &ptx_size));

            ptx.resize(ptx_size + 1);
            nvrtc_check(nvrtcGetPTX(program, &ptx[0]));
        }

        nvrtcDestroyProgram(&program);
        return result == NVRTC_SUCCESS;
    } catch (std::exception&) {
        nvrtcDestroyProgram(&program);
        throw;
    }
}

static bool extract_include_info_from_compile_error(
    std::string log,
    std::string& name,
    std::string& parent) {
    static const std::vector<std::string> pattern = {
        "could not open source file \"",
        "cannot open source file \""};

    for (auto& p : pattern) {
        size_t beg;
        size_t end;

        if ((beg = log.find(p)) == std::string::npos) {
            continue;
        }

        beg += p.size();

        if ((end = log.find("\"", beg)) == std::string::npos) {
            continue;
        }

        name = log.substr(beg, end - beg);

        size_t line_beg = log.rfind("\n", beg);
        line_beg = line_beg == std::string::npos ? 0 : line_beg + 1;
        size_t split = log.find("(", line_beg);
        parent = log.substr(line_beg, split - line_beg);

        return true;
    }

    return false;
}

static std::string generate_expression(
    const std::string& kernel_name,
    const std::vector<TemplateArg>& template_args,
    const std::vector<TypeInfo>& parameter_types) {
    std::stringstream oss;
    oss << "(void(*)(";

    bool is_first = true;
    for (const TypeInfo& ty : parameter_types) {
        if (!is_first) {
            oss << ",";
        } else {
            is_first = false;
        }

        oss << ty.name();
    }

    oss << "))";
    oss << kernel_name;

    if (!template_args.empty()) {
        oss << "<";

        is_first = true;
        for (const TemplateArg& arg : template_args) {
            if (!is_first) {
                oss << ",";
            } else {
                is_first = false;
            }

            oss << arg.get();
        }

        oss << ">";
    }

    return oss.str();
}

CudaModule NvrtcCompiler::compile(const KernelDef& spec) const {
    std::string lowered_name;
    std::string ptx;
    compile_ptx(spec, ptx, lowered_name);
    return {ptx.c_str(), lowered_name.c_str()};
}

static void add_std_flag(std::vector<std::string>& options) {
    for (const std::string& opt : options) {
        if (opt.find("--std") == 0 || opt.find("-std") == 0) {
            return;
        }
    }

    options.emplace_back("--std=c++11");
}

static void
add_arch_flag(std::vector<std::string>& options, CudaDevice device) {
    int minor = device.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    int major = device.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);

    std::stringstream oss;
    oss << "--gpu-architecture=compute_" << major << minor;
    options.push_back(oss.str());
}

void NvrtcCompiler::compile_ptx(
    const KernelDef& def,
    std::string& ptx,
    std::string& symbol_name,
    CudaDevice device) const {
    std::string symbol = generate_expression(
        def.kernel_name,
        def.template_args,
        def.param_types);
    std::string source = def.source.read(fs_);
    std::string log;

    bool mentions_std = false;
    std::vector<std::string> options;
    for (const std::string& opt : default_options_) {
        options.push_back(opt);
    }

    for (const std::string& opt : def.options) {
        options.push_back(opt);
    }

    add_std_flag(options);
    add_arch_flag(options, device);

    std::vector<const char*> raw_options;
    for (const std::string& opt : options) {
        raw_options.push_back(opt.c_str());
    }

    std::vector<const char*> headers_names;
    std::vector<const char*> headers_content;

    while (true) {
        headers_names.clear();
        headers_content.clear();

        for (const auto& p : file_cache_) {
            headers_names.push_back(p.first.c_str());
            headers_content.push_back(p.second.c_str());
        }

        bool success = nvrtc_compile(
            source,
            def.source.file_name(),
            symbol,
            raw_options,
            headers_names,
            headers_content,
            symbol_name,
            ptx,
            log);

        if (success) {
            break;
        }

        std::string missing_file;
        std::string parent_file;
        if (extract_include_info_from_compile_error(
                log,
                missing_file,
                parent_file)) {
            break;
        }

        std::vector<char> content = fs_.read(missing_file);

        file_cache_.emplace_back(
            std::move(missing_file),
            std::string(content.begin(), content.end()));
    }

    throw NvrtcException("NVRTC compilation failed: " + log);
}

int NvrtcCompiler::version() {
    int major, minor;
    if (nvrtcVersion(&major, &minor) == NVRTC_SUCCESS) {
        return 1000 * major + 10 * minor;
    } else {
        return 0;
    }
}

}  // namespace kernel_launcher