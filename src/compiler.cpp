#if defined(__HIP_PLATFORM_AMD__)
    #include <hip/hiprtc.h>
#else
    #include <nvrtc.h>
#endif
#include <unistd.h>

#include <cstring>
#include <deque>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

#include "kernel_launcher/kernel_builder.hpp"

#if defined(__HIP_PLATFORM_AMD__)
    #define nvrtcResult             hiprtcResult
    #define nvrtcProgram            hiprtcProgram
    #define NVRTC_SUCCESS           HIPRTC_SUCCESS
    #define NVRTC_ERROR_COMPILATION HIPRTC_ERROR_COMPILATION
    #define nvrtcGetErrorString     hiprtcGetErrorString
    #define nvrtcVersion            hiprtcVersion
    #define nvrtcCreateProgram      hiprtcCreateProgram
    #define nvrtcDestroyProgram     hiprtcDestroyProgram
    #define nvrtcAddNameExpression  hiprtcAddNameExpression
    #define nvrtcCompileProgram     hiprtcCompileProgram
    #define nvrtcGetProgramLogSize  hiprtcGetProgramLogSize
    #define nvrtcGetProgramLog      hiprtcGetProgramLog
    #define nvrtcGetLoweredName     hiprtcGetLoweredName
    #define nvrtcGetCUBINSize       hiprtcGetCodeSize
    #define nvrtcGetCUBIN           hiprtcGetCode
#endif

namespace kernel_launcher {

// Defined in jitsafe_headers.cpp. Maps header names (e.g. "stdint.h") to
// replacements that are safe to compile with NVRTC.
const std::unordered_map<std::string, std::string>& jitsafe_headers();

static void nvrtc_check(nvrtcResult result) {
    if (result != NVRTC_SUCCESS) {
        throw CompilerException(
            std::string("runtime compiler error: ") + nvrtcGetErrorString(result));
    }
}

int RuntimeCompiler::version() {
    int major = 0;
    int minor = 0;
    nvrtc_check(nvrtcVersion(&major, &minor));

    // For NVRTC, this is the same encoding as `CUDA_VERSION`, e.g. 12020 for 12.2.
    return major * 1000 + minor * 10;
}

// RAII wrapper around a `nvrtcProgram` handle; destroys it automatically.
struct NvrtcProgramDestroyer {
    nvrtcProgram program = nullptr;

    nvrtcProgram* ptr() {
        return &program;
    }

    operator nvrtcProgram() const {
        return program;
    }

    ~NvrtcProgramDestroyer() {
        if (program != nullptr) {
            nvrtcDestroyProgram(&program);
        }
    }
};

FileSource::FileSource(std::string path) : path_(std::move(path)) {}

FileSource::FileSource(std::string path, std::string content) :
    path_(std::move(path)),
    content_(std::move(content)) {}

std::string FileSource::read(const FileLoader& loader) const {
    if (content_) {
        return *content_;
    }

    return loader.load(path_);
}

size_t FileSource::hash() const noexcept {
    return hash_fields(path_, content_);
}

bool FileSource::operator==(const FileSource& that) const {
    return path_ == that.path_ && content_ == that.content_;
}

size_t KernelDef::calculate_hash() const noexcept {
    return hash_fields(kernel_name, source, headers, param_types, template_args, compiler_flags);
}

bool KernelDef::operator==(const KernelDef& that) const {
    return kernel_name == that.kernel_name && source == that.source && headers == that.headers
        && param_types == that.param_types && template_args == that.template_args
        && compiler_flags == that.compiler_flags;
}

static bool nvrtc_compile(
    const std::string& kernel_source,
    std::string_view kernel_file,
    const std::string& symbol_name,
    const std::vector<std::string>& options,
    const std::unordered_map<std::string, std::string>& headers,
    std::string& lowered_name,
    std::string& image,
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

    // NVRTC needs a null-terminated name, which a string_view does not
    // guarantee.
    std::string kernel_file_name(kernel_file);

    NvrtcProgramDestroyer program;
    nvrtc_check(nvrtcCreateProgram(
        program.ptr(),
        kernel_source.c_str(),
        kernel_file_name.c_str(),
        static_cast<int>(num_headers),
        headers_content.data(),
        headers_names.data()));

    nvrtc_check(nvrtcAddNameExpression(program, symbol_name.data()));
    nvrtcResult result = nvrtcCompileProgram(program, (int)options.size(), raw_options.data());

    if (result != NVRTC_SUCCESS && result != NVRTC_ERROR_COMPILATION) {
        nvrtc_check(result);
    }

    size_t log_size = 0;
    nvrtc_check(nvrtcGetProgramLogSize(program, &log_size));
    log.resize(log_size);
    nvrtc_check(nvrtcGetProgramLog(program, &log[0]));

    // The log size includes the trailing null terminator.
    while (!log.empty() && log.back() == '\0') {
        log.pop_back();
    }

    if (result != NVRTC_SUCCESS) {
        return false;
    }

    const char* ptr = nullptr;
    nvrtc_check(nvrtcGetLoweredName(program, symbol_name.data(), &ptr));
    lowered_name = ptr;

    size_t image_size = 0;
    nvrtc_check(nvrtcGetCUBINSize(program, &image_size));

    image.resize(image_size);
    nvrtc_check(nvrtcGetCUBIN(program, &image[0]));

    return true;
}

static std::string
generate_display_name(std::string_view kernel_name, const std::vector<TemplateArg>& template_args) {
    std::stringstream oss;
    oss << kernel_name;

    if (!template_args.empty()) {
        oss << "<";

        bool is_first = true;
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

static std::string generate_expression(
    std::string_view kernel_name,
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
    oss << generate_display_name(kernel_name, template_args);
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

// NVRTC has its own flags, while HIPRTC forwards its options to clang.
static void add_arch_flag(std::vector<std::string>& options, GPUArch arch) {
#if defined(__HIP_PLATFORM_AMD__)
    options.push_back("--offload-arch=" + arch.get());
#else
    options.push_back("--gpu-architecture=" + arch.get());
#endif
}

static std::string include_dir_flag(const std::string& dir) {
#if defined(__HIP_PLATFORM_AMD__)
    return "-I" + dir;
#else
    return "--include-path=" + dir;
#endif
}

static void add_pre_include_flag(std::vector<std::string>& options, const std::string& path) {
#if defined(__HIP_PLATFORM_AMD__)
    options.emplace_back("-include");
    options.push_back(path);
#else
    options.push_back("--pre-include=" + path);
#endif
}

static void add_default_device_flag(std::vector<std::string>& options) {
#if defined(__HIP_PLATFORM_AMD__)
    // This flag only exists for NVRTC.
    (void)options;
#else
    for (const std::string& opt : options) {
        if (opt == "--device-as-default-execution-space" || opt == "-default-device") {
            return;
        }
    }

    options.emplace_back("--device-as-default-execution-space");
#endif
}

static std::vector<std::string> extract_include_dirs(const std::vector<std::string>& options) {
    std::vector<std::string> result;

    for (size_t i = 0; i < options.size(); i++) {
        if (options[i].rfind("--include-path=", 0) == 0) {
            result.emplace_back(options[i], strlen("--include-path="));
        } else if (options[i] == "-I") {
            if (i + 1 >= options.size()) {
                throw std::runtime_error("missing argument for -I");
            }
            result.push_back(options[++i]);
        } else if (options[i].rfind("-I", 0) == 0) {
            result.emplace_back(options[i], strlen("-I"));
        }
    }

    return result;
}

static bool extract_unknown_header_from_log(std::string_view log, std::string& filename_out) {
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

    // Clang (used by HIPRTC) reports: fatal error: 'foo.h' file not found
    size_t end = log.find("' file not found");
    if (end != std::string::npos && end > 0) {
        size_t begin = log.rfind('\'', end - 1);

        if (begin != std::string::npos && begin + 1 < end) {
            filename_out = log.substr(begin + 1, end - begin - 1);
            return true;
        }
    }

    return false;
}

struct TempFile {
    explicit TempFile(std::string_view content) {
        const std::string& prefix = "kernel_launcher_header_";
        const std::string& suffix = ".cuh";
        path = (std::filesystem::temp_directory_path() / (prefix + "XXXXXX" + suffix)).string();

        int fd = ::mkstemps(&path[0], (int)suffix.size());
        if (fd < 0) {
            throw std::runtime_error("failed to create temporary file: " + path);
        }

        ssize_t result = ::write(fd, content.data(), content.size());
        ::close(fd);

        if (result != static_cast<ssize_t>(content.size())) {
            // The destructor does not run if the constructor throws.
            ::unlink(path.c_str());
            throw std::runtime_error("failed to write temporary file: " + path);
        }
    }

    ~TempFile() {
        ::unlink(path.c_str());
    }

    std::string path;
};

void RuntimeCompiler::compile_image(
    GPUArch arch,
    KernelDef def,
    std::string& image_out,
    std::string& symbol_out) const {
    constexpr size_t max_attempts = 25;

    std::string expression =
        generate_expression(def.kernel_name, def.template_args, def.param_types);
    std::string log;

    std::vector<std::string> options = default_options_;
    options.insert(options.end(), def.compiler_flags.begin(), def.compiler_flags.end());

    add_std_flag(options);
    add_arch_flag(options, arch);
    add_default_device_flag(options);

    // Before the user's flags, so that these directories take priority over
    // explicit -I directories. This allows overriding files for debugging.
    std::vector<std::string> env_options;
    for (const auto& dir : include_dirs_from_env()) {
        env_options.push_back(include_dir_flag(dir));
    }
    options.insert(options.begin(), env_options.begin(), env_options.end());

    std::vector<std::string> dirs = extract_include_dirs(options);
    dirs.insert(dirs.end(), search_dirs_.begin(), search_dirs_.end());
    DefaultLoader local_fs(std::move(dirs), loader_);

    std::string source = def.source.read(local_fs);
    std::vector<std::unique_ptr<TempFile>> preheaders;

    for (const auto& preheader : def.headers) {
        preheaders.emplace_back(std::make_unique<TempFile>(preheader.read(local_fs)));
        add_pre_include_flag(options, preheaders.back()->path);
    }

    log_debug() << "compiling " << def.kernel_name << " (" << def.source.file_name()
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
            image_out,
            log);

        log_debug() << "runtime compilation of " << def.source.file_name() << ": " << log
                    << std::endl;

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
        // If it cannot be found, retry with an empty header.
        std::string header_content;
        if (auto content = local_fs.try_load(header_name)) {
            header_content = std::move(*content);
        } else {
            log_debug() << "header " << header_name << " not found, retrying with an empty header"
                        << std::endl;
        }

        // HIPRTC rejects headers without content.
        if (header_content.empty()) {
            header_content = "\n";
        }

        headers.emplace(std::move(header_name), std::move(header_content));
    }

    throw CompilerException("runtime compilation failed: " + log);
}

std::unique_ptr<GPUKernel> RuntimeCompiler::compile(GPUContextRef ctx, KernelDef def) {
    std::string display_name = generate_display_name(def.kernel_name, def.template_args);
    std::string image;
    std::string symbol;
    compile_image(GPUArch::of(ctx.device()), std::move(def), image, symbol);

    // The module is loaded into the current context, so make `ctx` current.
    GPUContextGuard guard(ctx);
    return GPUKernel::load_image(image.data(), symbol, std::move(display_name));
}

}  // namespace kernel_launcher
