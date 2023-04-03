#include "kernel_launcher/compiler.h"

#include <nvrtc.h>
#include <unistd.h>

#include <cstring>
#include <sstream>

namespace kernel_launcher {

extern const std::unordered_map<std::string, std::string>& jitsafe_headers();

std::string KernelSource::read(const FileLoader& fs) const {
    if (!has_content_) {
        return fs.load(filename_);
    }

    return content_;
}

KernelDef::KernelDef(std::string name, KernelSource source) :
    name(std::move(name)),
    source({source}) {
    add_compiler_option("-DKERNEL_LAUNCHER=1");
    add_compiler_option("-Dkernel_tuner=1");
}

void KernelDef::add_template_arg(TemplateArg arg) {
    template_args.push_back(std::move(arg));
}

void KernelDef::add_parameter(TypeInfo dtype) {
    param_types.push_back(std::move(dtype));
}

void KernelDef::add_preincluded_header(KernelSource s) {
    preheaders.push_back(std::move(s));
}

void KernelDef::add_compiler_option(std::string option) {
    options.push_back(std::move(option));
}

CudaModule ICompiler::compile(CudaContextHandle ctx, KernelDef def) const {
    std::string human_name = def.name;
    std::string lowered_name;
    std::string ptx;
    compile_ptx(std::move(def), ctx.device().arch(), ptx, lowered_name);
    return {ptx.c_str(), lowered_name.c_str(), human_name.c_str()};
}

void Compiler::compile_ptx(
    KernelDef def,
    CudaArch arch,
    std::string& ptx_out,
    std::string& symbol_out) const {
    if (!inner_) {
        throw std::runtime_error(
            "kernel_launcher::Compiler has not been initialized");
    }

    return inner_->compile_ptx(std::move(def), arch, ptx_out, symbol_out);
}

CudaModule Compiler::compile(CudaContextHandle ctx, KernelDef def) const {
    if (!inner_) {
        throw std::runtime_error(
            "kernel_launcher::Compiler has not been initialized");
    }

    return inner_->compile(ctx, std::move(def));
}

static void nvrtc_check(nvrtcResult result) {
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
        if (program_ != nullptr) {
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
    const std::vector<std::string>& options,
    const std::unordered_map<std::string, std::string>& headers,
    std::string& lowered_name,
    std::string& ptx,
    std::string& log) {
    std::vector<const char*> raw_options;
    for (const auto& opt : options) {
        raw_options.push_back(opt.c_str());
    }

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
        nvrtcCompileProgram(program, (int)options.size(), raw_options.data());

    if (result != NVRTC_SUCCESS && result != NVRTC_ERROR_COMPILATION) {
        nvrtc_check(result);
    }

    size_t log_size = 0;
    nvrtc_check(nvrtcGetProgramLogSize(program, &log_size));
    log.resize(log_size + 1);
    nvrtc_check(nvrtcGetProgramLog(program, &log[0]));

    if (result != NVRTC_SUCCESS) {
        return false;
    }

    const char* ptr = nullptr;
    nvrtc_check(nvrtcGetLoweredName(program, symbol_name.data(), &ptr));
    lowered_name = ptr;

    size_t ptx_size = 0;
    nvrtc_check(nvrtcGetPTXSize(program, &ptx_size));

    ptx.resize(ptx_size + 1);
    nvrtc_check(nvrtcGetPTX(program, &ptx[0]));

    return true;
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

static void add_std_flag(std::vector<std::string>& options) {
    for (const std::string& opt : options) {
        if (opt.find("--std") == 0 || opt.find("-std") == 0) {
            return;
        }
    }

    options.emplace_back("--std=c++11");
}

static void add_arch_flag(std::vector<std::string>& options, CudaArch arch) {
    std::stringstream oss;
    oss << "--gpu-architecture=compute_" << arch.get();
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

struct TempFile {
    explicit TempFile(const std::string& content) {
        const std::string& prefix = "header_";
        const std::string& suffix = ".cuh";
        path = prefix + "XXXXXX" + suffix;

        int fd = ::mkstemps(&path[0], (int)suffix.size());
        if (fd < 0) {
            throw std::runtime_error(
                "failed to create temporary file: " + path);
        }

        ssize_t result = ::write(fd, content.c_str(), content.size());
        ::close(fd);

        if (result != static_cast<ssize_t>(content.size())) {
            throw std::runtime_error("failed to write temporary file: " + path);
        }
    }

    ~TempFile() {
        if (!path.empty()) {
            unlink(path.c_str());
            path = "";
        }
    }

    std::string path;
};

void NvrtcCompiler::compile_ptx(
    KernelDef def,
    CudaArch arch,
    std::string& ptx_out,
    std::string& symbol_out) const {
    constexpr size_t max_attempts = 25;

    std::string expression =
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
    add_arch_flag(options, arch);
    add_default_device_flag(options);

    std::vector<std::string> dirs = extract_include_dirs(options);
    ForwardLoader local_fs = ForwardLoader(std::move(dirs), fs_);

    std::string source = def.source.read(local_fs);
    std::vector<TempFile> preheaders;

    for (const auto& preheader : def.preheaders) {
        preheaders.emplace_back(preheader.read(local_fs));
        options.push_back("--pre-include=" + preheaders.back().path);
    }

    log_debug() << "compiling " << def.name << " (" << def.source.file_name()
                << "): " << source << "\n";

    std::unordered_map<std::string, std::string> headers;
    for (size_t attempt = 0; attempt < max_attempts; attempt++) {
        bool success = nvrtc_compile(
            source,
            def.source.file_name(),
            expression,
            options,
            headers,
            symbol_out,
            ptx_out,
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
        std::string header_content;
        try {
            header_content = local_fs.load(header_name);
        } catch (const std::exception& e) {
            log_debug() << "retrying compilation after error: " << e.what()
                        << std::endl;
        }

        headers.emplace(std::move(header_name), std::move(header_content));
    }

    throw NvrtcException("NVRTC compilation failed: " + log);
}

int NvrtcCompiler::version() {
    int major;
    int minor;

    if (nvrtcVersion(&major, &minor) != NVRTC_SUCCESS) {
        return 0;
    }

    return 1000 * major + 10 * minor;
}

Compiler global_compiler;

void set_global_default_compiler(Compiler c) {
    global_compiler = std::move(c);
}

Compiler default_compiler() {
    if (!global_compiler.is_initialized()) {
        global_compiler = NvrtcCompiler {};
    }

    return global_compiler;
}

}  // namespace kernel_launcher