#include "kernel_launcher/gpu_utils.hpp"

#include <cuda_runtime_api.h>
#include <cxxabi.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "kernel_launcher/utils.hpp"

#if defined(__HIP_PLATFORM_AMD__)
    #define cuGetErrorName            hipDrvGetErrorName
    #define cuGetErrorString          hipDrvGetErrorString
    #define cuMemcpyAsync             hipMemcpyAsync  // NOTE: HIP takes an extra `hipMemcpyKind` argument
    #define cuCtxSynchronize          hipCtxSynchronize
    #define cuStreamSynchronize       hipStreamSynchronize
    #define cuDriverGetVersion        hipDriverGetVersion
    #define cuModuleGetFunction       hipModuleGetFunction
    #define cuModuleLoadDataEx        hipModuleLoadDataEx
    #define cuModuleUnload            hipModuleUnload
    #define cuLaunchKernel            hipModuleLaunchKernel
    #define cuLaunchCooperativeKernel hipModuleLaunchCooperativeKernel
    #define cuFuncGetAttribute        hipFuncGetAttribute
    #define cuOccupancyMaxActiveBlocksPerMultiprocessor \
        hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    #define cuOccupancyMaxPotentialBlockSize \
        hipModuleOccupancyMaxPotentialBlockSize  // NOTE: HIP has no block-to-smem callback argument
    #define cuInit               hipInit
    #define cuDeviceGetCount     hipGetDeviceCount
    #define cuCtxGetDevice       hipCtxGetDevice
    #define cuDeviceGetName      hipDeviceGetName
    #define cuDeviceGetAttribute hipDeviceGetAttribute
    // cuCtxGetId: no HIP equivalent
    #define cuCtxGetCurrent  hipCtxGetCurrent
    #define cuCtxPushCurrent hipCtxPushCurrent
    #define cuCtxPopCurrent  hipCtxPopCurrent
#endif

namespace kernel_launcher {

std::string message_from_curesult(GPUResult error_code, std::string_view msg = "") {
    const char* name = "unknown";
    const char* string = "unknown error";

    cuGetErrorName(error_code, &name);
    cuGetErrorString(error_code, &string);

    std::stringstream stream;

    if (!msg.empty()) {
        stream << msg << " (";
    }

    stream << name << " [" << error_code << "]: " << string;

    if (!msg.empty()) {
        stream << ")";
    }

    return stream.str();
}

GPUException::GPUException(GPUResult error_code) :
    std::runtime_error(message_from_curesult(error_code)),
    error_code_(error_code) {}

GPUException::GPUException(GPUResult error_code, std::string_view msg) :
    std::runtime_error(message_from_curesult(error_code, msg)),
    error_code_(error_code) {}

void gpu_copy(void* dst, const void* src, size_t nbytes, GPUStream stream) {
    GPUResult result = cuMemcpyAsync(
        static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(dst)),
        static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(src)),
        nbytes,
        stream);

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuMemcpyAsync failed");
    }
}

void gpu_synchronize(GPUStream stream) {
    if (stream == nullptr) {
        GPUResult result = cuCtxSynchronize();

        if (result != CUDA_SUCCESS) {
            throw GPUException(result, "cuCtxSynchronize failed");
        }
    } else {
        GPUResult result = cuStreamSynchronize(stream);

        if (result != CUDA_SUCCESS) {
            throw GPUException(result, "cuStreamSynchronize failed");
        }
    }
}

int gpu_driver_version() {
    int version = 0;
    GPUResult result = cuDriverGetVersion(&version);

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuDriverGetVersion failed");
    }

    return version;
}

int gpu_runtime_version() {
    int version = 0;
    cudaError_t result = cudaRuntimeGetVersion(&version);

    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaRuntimeGetVersion failed: ") + cudaGetErrorString(result));
    }

    return version;
}

static GPUFunction get_module_function(GPUModule module, const std::string& symbol_name) {
    GPUFunction function;
    GPUResult result = cuModuleGetFunction(&function, module, symbol_name.c_str());

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuModuleGetFunction failed: " + symbol_name);
    }

    return function;
}

std::unique_ptr<GPUKernel> GPUKernel::load_image(
    const void* image,
    const std::string& symbol_name,
    std::string display_name,
    const std::vector<std::pair<CUjit_option, void*>>& options) {
    std::vector<char> error_log;
    static constexpr size_t error_log_capacity = 8192;

    bool has_error_log_buffer = false;
    for (const auto& option : options) {
        if (option.first == CU_JIT_ERROR_LOG_BUFFER) {
            has_error_log_buffer = true;
        }
    }

    std::vector<CUjit_option> option_keys;
    std::vector<void*> option_values;
    option_keys.reserve(options.size() + 2);
    option_values.reserve(options.size() + 2);

    if (!has_error_log_buffer) {
        error_log.resize(error_log_capacity + 1, '\0');

        option_keys.push_back(CU_JIT_ERROR_LOG_BUFFER);
        option_values.push_back(error_log.data());
        option_keys.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
        option_values.push_back(reinterpret_cast<void*>(error_log.size()));
    }

    for (const auto& option : options) {
        option_keys.push_back(option.first);
        option_values.push_back(option.second);
    }

    GPUModule module;
    GPUResult result = cuModuleLoadDataEx(
        &module,
        image,
        static_cast<unsigned int>(option_keys.size()),
        option_keys.data(),
        option_values.data());

    if (result != CUDA_SUCCESS) {
        std::string msg = "cuModuleLoadDataEx failed";
        size_t log_len = strnlen(error_log.data(), error_log.size());

        if (log_len > 0) {
            msg += ": " + std::string(error_log.data(), log_len);
        }

        throw GPUException(result, msg);
    }

    return std::make_unique<GPUKernel>(module, symbol_name, std::move(display_name));
}

GPUKernel::GPUKernel(GPUModule module, GPUFunction function, std::string display_name) :
    module_(module),
    function_(function),
    name_(std::move(display_name)) {}

GPUKernel::GPUKernel(
    GPUModule module,
    const std::string& symbol_name,
    std::string display_name) try :
    GPUKernel(module, get_module_function(module, symbol_name), display_name) {
} catch (...) {
    cuModuleUnload(module);
    throw;
}

GPUKernel::~GPUKernel() {
    if (module_ != nullptr) {
        cuModuleUnload(module_);
    }
}

void GPUKernel::launch(GPUStream stream, LaunchConfig config, const void* const* args) const {
    GPUResult result;

    if (config.grid_size[0] * config.grid_size[1] * config.grid_size[2] == 0) {
        log_debug() << "skipping kernel launch as grid size is 0\n";
        return;
    }

    // cuLaunchKernel/cuLaunchCooperativeKernel take `void**` even though they
    // only read through it. Cast is required to remove const.
    void** mutable_args = const_cast<void**>(args);

    if (config.cooperative) {
        result = cuLaunchCooperativeKernel(
            function_,
            config.grid_size[0],
            config.grid_size[1],
            config.grid_size[2],
            config.block_size[0],
            config.block_size[1],
            config.block_size[2],
            config.shared_mem,
            stream,
            mutable_args);
    } else {
        result = cuLaunchKernel(
            function_,
            config.grid_size[0],
            config.grid_size[1],
            config.grid_size[2],
            config.block_size[0],
            config.block_size[1],
            config.block_size[2],
            config.shared_mem,
            stream,
            mutable_args,
            nullptr);
    }

    if (result != CUDA_SUCCESS) {
        std::string msg =
            config.cooperative ? "cuLaunchCooperativeKernel failed" : "cuLaunchKernel failed";
        throw GPUException(result, msg + " for kernel '" + name_ + "'");
    }
}

int GPUKernel::attribute(CUfunction_attribute attr) const {
    int value = 0;
    GPUResult result = cuFuncGetAttribute(&value, attr, function_);

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuFuncGetAttribute failed for kernel '" + name_ + "'");
    }

    return value;
}

int GPUKernel::max_threads_per_block() const {
    return attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

int GPUKernel::num_regs() const {
    return attribute(CU_FUNC_ATTRIBUTE_NUM_REGS);
}

int GPUKernel::max_active_blocks_per_sm(uint32_t block_size, uint32_t dynamic_shared_mem) const {
    int value = 0;
    GPUResult result = cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &value,
        function_,
        static_cast<int>(block_size),
        static_cast<size_t>(dynamic_shared_mem));

    if (result != CUDA_SUCCESS) {
        throw GPUException(
            result,
            "cuOccupancyMaxActiveBlocksPerMultiprocessor failed for kernel '" + name_ + "'");
    }

    return value;
}

int GPUKernel::max_active_blocks(uint32_t block_size, uint32_t dynamic_shared_mem, GPUDevice device)
    const {
    int num_sms = device.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    return num_sms * max_active_blocks_per_sm(block_size, dynamic_shared_mem);
}

GPUKernel::BlockSizeHint
GPUKernel::optimal_threads_per_block(size_t dynamic_shared_mem, int block_size_limit) const {
    int min_grid_size = 0;
    int block_size = 0;

    GPUResult result = cuOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        function_,
        nullptr,
        dynamic_shared_mem,
        block_size_limit);

    if (result != CUDA_SUCCESS) {
        throw GPUException(
            result,
            "cuOccupancyMaxPotentialBlockSize failed for kernel '" + name_ + "'");
    }

    return {min_grid_size, block_size};
}

int GPUDevice::count() {
    // The driver API must be initialized before any other call. `cuInit` is
    // cheap after the first successful call. It fails with
    // `CUDA_ERROR_NO_DEVICE` if no GPUs are available (including when
    // `CUDA_VISIBLE_DEVICES` is empty).
    GPUResult result = cuInit(0);

    if (result == CUDA_ERROR_NO_DEVICE) {
        return 0;
    }

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuInit failed");
    }

    int value = 0;
    result = cuDeviceGetCount(&value);

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuDeviceGetCount failed");
    }

    return value;
}

GPUDevice GPUDevice::current() {
    CUdevice device;
    GPUResult result = cuCtxGetDevice(&device);

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuCtxGetDevice failed");
    }

    return GPUDevice(device);
}

GPUArch GPUArch::of(GPUDevice device) {
#if defined(__HIP_PLATFORM_AMD__)
    hipDeviceProp_t props;
    GPUResult result = hipGetDeviceProperties(&props, device.get());

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "hipGetDeviceProperties failed");
    }

    // The name may be followed by target features (e.g. "gfx90a:sramecc+:xnack-").
    std::string arch = props.gcnArchName;
    arch = arch.substr(0, arch.find(':'));
#else
    int major = device.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    int minor = device.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    std::string arch = "sm_" + std::to_string(major) + std::to_string(minor);
#endif

    return GPUArch(std::move(arch), device.name());
}

GPUArch GPUArch::current() {
    return GPUArch::of(GPUDevice::current());
}

std::string GPUDevice::name() const {
    char buffer[256];
    GPUResult result = cuDeviceGetName(buffer, sizeof(buffer), static_cast<CUdevice>(device_));

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuDeviceGetName failed");
    }

    return std::string(buffer);
}

int GPUDevice::attribute(CUdevice_attribute attr) const {
    int value = 0;
    GPUResult result = cuDeviceGetAttribute(&value, attr, static_cast<CUdevice>(device_));

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuDeviceGetAttribute failed");
    }

    return value;
}

GPUContextRef::GPUContextRef(CUcontext context) : context_(context) {
    // Checked explicitly, since `cuCtxGetId` would silently use the current
    // context for a null handle.
    if (context == nullptr) {
        throw std::invalid_argument("GPUContextRef: context must not be null");
    }

#if CUDA_VERSION >= 12000
    GPUResult result = cuCtxGetId(context, &id_);

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuCtxGetId failed");
    }
#else
    // cuCtxGetId is only available since CUDA 12.0. On older toolkits, fall
    // back to the context handle itself: it is unique within this process
    // for as long as the context is alive.
    id_ = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(context));
#endif
}

GPUContextRef GPUContextRef::current() {
    CUcontext ctx;
    GPUResult result = cuCtxGetCurrent(&ctx);

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuCtxGetCurrent failed");
    }

    if (ctx == nullptr) {
        throw std::runtime_error("no current CUDA context is set on this thread");
    }

    return GPUContextRef(ctx);
}

unsigned long long GPUContextRef::id() const {
    return id_;
}

GPUDevice GPUContextRef::device() const {
    CUcontext current;
    GPUResult result = cuCtxGetCurrent(&current);

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuCtxGetCurrent failed");
    }

    // If this context is already current, no need to push/pop it.
    if (current == context_) {
        return GPUDevice::current();
    }

    GPUContextGuard guard(context_);
    return GPUDevice::current();
}

GPUContextGuard::GPUContextGuard(CUcontext context) {
    GPUResult result = cuCtxPushCurrent(context);

    if (result != CUDA_SUCCESS) {
        throw GPUException(result, "cuCtxPushCurrent failed");
    }
}

GPUContextGuard::~GPUContextGuard() {
    // Errors are ignored since destructors must not throw.
    CUcontext popped;
    cuCtxPopCurrent(&popped);
}

}  // namespace kernel_launcher
