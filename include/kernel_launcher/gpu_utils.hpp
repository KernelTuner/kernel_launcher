#ifndef KERNEL_LAUNCHER_GPU_UTILS_H
#define KERNEL_LAUNCHER_GPU_UTILS_H

#if defined(__HIP_PLATFORM_AMD__)
    #include <hip/hip_runtime.h>
#else
    #include <cuda.h>
#endif

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kernel_launcher {

/**
 * @addtogroup gpu
 * @{
 */

#if defined(__HIP_PLATFORM_AMD__)
using GPUResult = hipError_t;
using GPUFunction = hipFunction_t;
using GPUStream = hipStream_t;
using GPUModule = hipModule_t;
#else
using GPUResult = CUresult;
using GPUFunction = CUfunction;
using GPUStream = CUstream;
using GPUModule = CUmodule;
#endif

/**
 * Thrown when a GPU driver/runtime call (CUDA or HIP) returns an error.
 */
class GPUException: public std::runtime_error {
  public:
    GPUException(GPUResult error_code);
    GPUException(GPUResult error_code, std::string_view msg);

    GPUResult error_code() const {
        return error_code_;
    }

  private:
    GPUResult error_code_;
};

/**
 * Asynchronously copies `nbytes` from `src` to `dst` on the given stream. Both pointers
 * may refer to host or device memory; the direction is inferred via unified addressing.
 */
void gpu_copy(void* dst, const void* src, size_t nbytes, GPUStream stream = nullptr);

/**
 * Blocks until all work on `stream` has completed. If `stream` is null, blocks until all
 * work in the current context has completed.
 */
void gpu_synchronize(GPUStream stream = nullptr);

/**
 * Returns the version of the installed GPU driver (e.g. 12040 for CUDA 12.4).
 */
int gpu_driver_version();

/**
 * Returns the version of the GPU runtime library (e.g. 12040 for CUDA 12.4).
 */
int gpu_runtime_version();

/**
 * Wrapper around GPU device.
 */
class GPUDevice {
  public:
    GPUDevice() = default;

    /**
     * Constructs a `GPUDevice` from the given `CUdevice`.
     */
    explicit GPUDevice(CUdevice device) : device_(device) {}

    /**
     * Returns the number of available GPU devices.
     */
    static int count();

    /**
     * Returns a reference to the current active device.
     */
    static GPUDevice current();

    /**
     * Returns the number of this GPU device.
     */
    std::string name() const;

    /**
     * Returns the value of the give attribute.
     */
    int attribute(CUdevice_attribute attr) const;

    /**
     * Returns the underlying CUdevice.
     */
    CUdevice get() const {
        return device_;
    }

  private:
    CUdevice device_ = -1;
};

/**
 * GPU architecture identifier: a CUDA target name (e.g. "sm_80") or, on AMD, a
 * GCN/RDNA target name (e.g. "gfx90a").
 */
class GPUArch {
  public:
    GPUArch() = default;

    /**
     * Construct an architecture without querying a device. `name` is the device name, which is
     * only used to match wisdom records.
     */
    explicit GPUArch(std::string arch, std::string name = {}) :
        arch_(std::move(arch)),
        name_(std::move(name)) {}

    static GPUArch current();
    static GPUArch of(GPUDevice device);

    const std::string& get() const {
        return arch_;
    }

    const std::string& name() const {
        return name_;
    }

  private:
    std::string arch_;
    std::string name_;
};

/**
 * The configuration pass when launching a GPU kernel.
 */
struct LaunchConfig {
    uint32_t block_size[3] = {1, 1, 1};
    uint32_t grid_size[3] = {0, 1, 1};
    uint32_t shared_mem = 0;
    bool cooperative = false;
};

/**
 * Wrapper around a compiled kernel function plus its owning module.
 */
class GPUKernel {
  public:
    struct BlockSizeHint {
        int min_grid_size;
        int block_size;
    };

    /**
     * Loads the module `image` and looks up the kernel `symbol_name` (the mangled name). The
     * `display_name` is a human-readable name used in error messages. If empty, it defaults to
     * the demangled `symbol_name`.
     */
    static std::unique_ptr<GPUKernel> load_image(
        const void* image,
        const std::string& symbol_name,
        std::string display_name = {},
        const std::vector<std::pair<CUjit_option, void*>>& options = {});

    /**
     * Construct GPU module from the module object, function object, and display name. Note that
     * this class takes ownership, it will delete the module on destruction.
     */
    GPUKernel(GPUModule module, GPUFunction function, std::string display_name);

    /**
     * Construct GPU module from the module object, the kernel name inside the module, and an,
     * optional, display name. Note that this class takes ownership, it will delete the module on
     * destruction, even if that happens inside the constructor.
     */
    GPUKernel(GPUModule module, const std::string& symbol_name, std::string display_name = {});

    GPUKernel(const GPUKernel&) = delete;
    GPUKernel& operator=(const GPUKernel&) = delete;
    ~GPUKernel();

    /**
     * Launch this kernel on the given stream with the given arguments and configuration.
     */
    void launch(GPUStream stream, LaunchConfig config, const void* const* args) const;

    /**
     * Returns the number of registers consumed by this kernel.
     */
    int num_regs() const;

    /**
     * Returns the maximum number of active blocks per SM. The caller provides the block size
     * (i.e., threads per block) and the dynamic shared memory size (in bytes).
     */
    int max_active_blocks_per_sm(uint32_t block_size, uint32_t dynamic_shared_mem = 0) const;

    /**
     * Returns the maximum number of thread blocks that can be active on a device. The caller
     * provides the block size (i.e., threads per block), the dynamic  shared memory size
     * (in bytes), and the GPU device (defaults to the current device).
     *
     * This is simply calculated as:
     * `device.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) * max_active_blocks_per_sm`
     */
    int max_active_blocks(
        uint32_t block_size,
        uint32_t dynamic_shared_mem = 0,
        GPUDevice device = GPUDevice::current()) const;

    /**
     * Returns the maximum thread block size (i.e., thread per block).
     */
    int max_threads_per_block() const;

    /**
     * Returns the block size (i.e., threads per block) that maximizes occupancy for this
     * kernel, together with the minimum grid size needed to reach that occupancy on all
     * SMs. The caller provides the dynamic shared memory size (in bytes) and,
     * optionally, an upper bound on the block size (0 means no limit).
     *
     * This is a wrapper around `cuOccupancyMaxPotentialBlockSize`. The returned block size is
     * never larger than `max_threads_per_block()`.
     */
    BlockSizeHint
    optimal_threads_per_block(size_t dynamic_shared_mem = 0, int block_size_limit = 0) const;
    int attribute(CUfunction_attribute attr) const;

    GPUModule module() const {
        return module_;
    }

    GPUFunction function() const {
        return function_;
    }

    const std::string& name() const {
        return name_;
    }

  private:
    GPUModule module_;
    GPUFunction function_;
    std::string name_;
};

/**
 * Wrapper around a GPU context. Note that this object does not keep track of the lifetime of the
 * context, it is just a minimal wrapper around a CUcontext.
 */
class GPUContextRef {
  public:
    static GPUContextRef current();

    /**
     * Constructs a reference to the given `CUcontext`, which must not be
     * null. Does not take ownership of the context.
     */
    explicit GPUContextRef(CUcontext context);

    /**
     * The context id, as given by `cuCtxGetId`. This id is unique during the lifetime of
     * the program.
     */
    unsigned long long id() const;

    /**
     * The device associated with this context.
     */
    GPUDevice device() const;

    bool operator==(const GPUContextRef& that) const {
        return id() == that.id();
    }

    bool operator!=(const GPUContextRef& that) const {
        return !(*this == that);
    }

    CUcontext get() const {
        return context_;
    }

  private:
    CUcontext context_ = nullptr;
    unsigned long long id_ = 0;
};

/**
 * RAII guard that pushes the given context onto the current thread's context stack on
 * construction and pops it again on destruction.
 */
class GPUContextGuard {
  public:
    explicit GPUContextGuard(CUcontext context);
    explicit GPUContextGuard(GPUContextRef context) : GPUContextGuard(context.get()) {}

    GPUContextGuard(const GPUContextGuard&) = delete;
    GPUContextGuard& operator=(const GPUContextGuard&) = delete;

    ~GPUContextGuard();
};

// Deprecated: These are old names, here for backwards compatibility
using CudaException = GPUException;
using CudaModule = GPUKernel;
using CudaArch = GPUArch;
using CudaDevice = GPUDevice;
using CudaContextHandle = GPUContextRef;

/** @} */

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::GPUContextRef> {
    size_t operator()(const kernel_launcher::GPUContextRef& ctx) const noexcept {
        return std::hash<unsigned long long> {}(ctx.id());
    }
};
}  // namespace std

#endif
