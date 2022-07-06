#include "kernel_launcher/compiler.h"

#include <nvrtc.h>

#include <cstring>
#include <sstream>

namespace kernel_launcher {

extern const std::unordered_map<std::string, std::string>& jitsafe_headers();

std::string KernelSource::read(
    const FileLoader& fs,
    const std::vector<std::string>& dirs) const {
    if (!has_content_) {
        std::vector<char> content = fs.load(filename_, dirs);
        content.push_back('\0');
        return std::string(content.data());
    } else {
        return content_;
    }
}

KernelDef::KernelDef(std::string name, KernelSource source) :
    name(std::move(name)),
    source(source) {
    add_compiler_option("-DKERNEL_LAUNCHER=1");
    add_compiler_option("-Dkernel_tuner=1");
}

void KernelDef::add_template_arg(TemplateArg arg) {
    template_args.push_back(std::move(arg));
}

void KernelDef::add_parameter(TypeInfo dtype) {
    param_types.push_back(std::move(dtype));
}

void KernelDef::add_compiler_option(std::string option) {
    options.push_back(std::move(option));
}

inline void nvrtc_check(nvrtcResult result) {
    if (result != NVRTC_SUCCESS) {
        std::stringstream ss;
        ss << "NVRTC error: " << nvrtcGetErrorString(result);
        throw NvrtcException(ss.str());
    }
}

NvrtcCompiler::NvrtcCompiler(
    std::vector<std::string> options,
    std::shared_ptr<FileLoader> fs) :
    fs_(fs ? std::move(fs) : std::make_shared<DefaultLoader>()),
    default_options_(std::move(options)) {}

// RAII wrapper to ensure nvrtcDestroy is always called
struct NvrtcProgramDestroyer {
    ~NvrtcProgramDestroyer() {
        if (program_) {
            nvrtcDestroyProgram(&program_);
            program_ = nullptr;
        }
    }

    operator nvrtcProgram&() {
        return program_;
    }

  private:
    nvrtcProgram program_ = nullptr;
};

static bool nvrtc_compile(
    const std::string& kernel_source,
    const std::string& kernel_file,
    const std::string& symbol_name,
    const std::vector<const char*>& options,
    const std::unordered_map<std::string, std::string>& headers,
    std::string& lowered_name,
    std::string& ptx,
    std::string& log) {
    std::vector<const char*> headers_names;
    std::vector<const char*> headers_content;

    size_t num_headers = headers.size() + jitsafe_headers().size();
    headers_names.reserve(num_headers);
    headers_content.reserve(num_headers);

    for (const auto& p : headers) {
        headers_names.push_back(p.first.c_str());
        headers_content.push_back(p.second.c_str());
    }

    for (const auto& p : jitsafe_headers()) {
        headers_names.push_back(p.first.c_str());
        headers_content.push_back(p.second.c_str());
    }

    NvrtcProgramDestroyer program;
    nvrtc_check(nvrtcCreateProgram(
        &(nvrtcProgram&)program,
        kernel_source.c_str(),
        kernel_file.c_str(),
        static_cast<int>(num_headers),
        headers_content.data(),
        headers_names.data()));

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

    return result == NVRTC_SUCCESS;
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

CudaModule NvrtcCompiler::compile(CudaContextHandle ctx, KernelDef def) const {
    CudaDevice device = ctx.device();
    int minor = device.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    int major = device.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    int arch = major * 10 + minor;  // Is this always the case?

    std::string lowered_name;
    std::string ptx;
    compile_ptx(def, arch, ptx, lowered_name);
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

static void add_arch_flag(std::vector<std::string>& options, int arch_version) {
    std::stringstream oss;
    oss << "--gpu-architecture=compute_" << arch_version;
    options.push_back(oss.str());
}

static void add_default_device_flag(std::vector<std::string>& options) {
    for (const std::string& opt : options) {
        if (opt == "--device-as-default-execution-space"
            || opt == "-default-device") {
            return;
        }
    }

    options.emplace_back("--device-as-default-execution-space");
}

static std::vector<std::string>
extract_include_dirs(const std::vector<std::string>& options) {
    std::vector<std::string> result;

    for (size_t i = 0; i < options.size(); i++) {
        if (options[i].find("--include-path=") == 0) {
            result.emplace_back(options[i], strlen("--include-path="));
        }

        if (options[i].find("-I") == 0) {
            if (options[i].size() > strlen("-I")) {
                result.emplace_back(options[i], strlen("-I"));
            } else if (i < options.size()) {
                result.push_back(options[i + 1]);
            } else {
                //???
            }
        }
    }

    return result;
}

static bool extract_unknown_header_from_log(
    const std::string& log,
    std::string& filename_out) {
    static constexpr auto patterns = {
        "could not open source file \"",
        "cannot open source file \""};

    for (const char* pattern : patterns) {
        size_t begin = log.find(pattern);

        if (begin == std::string::npos) {
            continue;
        }

        begin += strlen(pattern);
        size_t end = log.find('\"', begin + 1);

        if (end == std::string::npos || end <= begin) {
            continue;
        }

        filename_out = log.substr(begin, end - begin);
        return true;
    }

    return false;
}

void NvrtcCompiler::compile_ptx(
    const KernelDef& def,
    int arch_version,
    std::string& ptx,
    std::string& symbol_name) const {
    constexpr size_t max_attempts = 25;

    std::string symbol =
        generate_expression(def.name, def.template_args, def.param_types);
    std::string log;

    std::vector<std::string> options;
    for (const std::string& opt : default_options_) {
        options.push_back(opt);
    }

    for (const std::string& opt : def.options) {
        options.push_back(opt);
    }

    add_std_flag(options);
    add_arch_flag(options, arch_version);
    add_default_device_flag(options);

    std::vector<const char*> raw_options;
    for (const std::string& opt : options) {
        raw_options.push_back(opt.c_str());
    }

    std::unordered_map<std::string, std::string> headers;
    std::vector<std::string> dirs = extract_include_dirs(options);
    std::string source = def.source.read(*fs_, dirs);

    log_debug() << "compiling " << def.name << " (" << def.source.file_name()
                << "): " << source << "\n";

    for (size_t attempt = 0; attempt < max_attempts; attempt++) {
        bool success = nvrtc_compile(
            source,
            def.source.file_name(),
            symbol,
            raw_options,
            headers,
            symbol_name,
            ptx,
            log);

        log_debug() << "NVRTC compilation of " << def.source.file_name() << ": "
                    << log << std::endl;

        if (success) {
            return;
        }

        // See if compilation failed due to missing header file
        std::string header_name;
        if (!extract_unknown_header_from_log(log, header_name)) {
            break;
        }

        // Header already loaded. Something is wrong?
        if (headers.count(header_name) > 0) {
            break;
        }

        // Load missing header file
        std::vector<char> header_content;
        try {
            header_content = fs_->load(header_name, dirs);
        } catch (const std::exception& e) {
            log_warning() << "retrying compilation after error: " << e.what()
                          << std::endl;
        }

        header_content.push_back('\0');
        headers.emplace(
            std::move(header_name),
            std::string(header_content.data()));
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