#include "catch.hpp"
#include "kernel_launcher/kernel_arg.hpp"

using namespace kernel_launcher;

namespace test_kernel_arg {
struct Custom {
    int a;
    float b;
};
}  // namespace test_kernel_arg

TEST_CASE("KernelArg scalar") {
    KernelArg arg = KernelArg::of(42);

    CHECK(arg.type() == TypeInfo::of<int>());
    CHECK(arg.has_type<int>());
    CHECK(arg.has_type<const int&>());
    CHECK_FALSE(arg.has_type<long>());

    CHECK(arg.get<int>() == 42);
    CHECK(*static_cast<const int*>(arg.data()) == 42);
    CHECK_THROWS(arg.get<long>());

    CHECK(arg.buffer_size() == 0);
    CHECK_THROWS(arg.with_buffer_size(10));

    CHECK(arg.to_primitive() == 42);
    CHECK(arg.to_primitive_or_empty() == 42);
    CHECK(KernelArg::of(2.5).to_primitive() == 2.5);
    CHECK(KernelArg::of(true).to_primitive() == true);
}

TEST_CASE("KernelArg custom type") {
    KernelArg arg = KernelArg::of(test_kernel_arg::Custom {1, 2.0f});

    CHECK(arg.has_type<test_kernel_arg::Custom>());
    CHECK(arg.get<test_kernel_arg::Custom>().a == 1);
    CHECK(arg.get<test_kernel_arg::Custom>().b == 2.0f);

    CHECK(arg.to_primitive_or_empty().is_empty());
    CHECK_THROWS(arg.to_primitive());
}

TEST_CASE("KernelArg pointer") {
    float data[4] = {0};
    KernelArg arg = KernelArg::of(data, sizeof(data));

    CHECK(arg.type() == TypeInfo::of<float*>());
    CHECK(arg.type().is_pointer());
    CHECK(arg.get<float*>() == data);
    CHECK(arg.buffer_size() == sizeof(data));
    CHECK(arg.to_primitive_or_empty().is_empty());

    // A `char*` points to GPU memory, so it is not converted to a string.
    char chars[4] = {'a', 'b', 'c', '\0'};
    CHECK(KernelArg::of(chars, sizeof(chars)).to_primitive_or_empty().is_empty());
    CHECK(KernelArg::of(static_cast<char*>(chars)).to_primitive_or_empty().is_empty());

    KernelArg resized = arg.with_buffer_size(8);
    CHECK(resized.buffer_size() == 8);
    CHECK(resized.get<float*>() == data);
    CHECK(arg.buffer_size() == sizeof(data));
}

TEST_CASE("KernelArg::unsafe_construct") {
    short value = 7;
    KernelArg arg = KernelArg::unsafe_construct(TypeInfo::of<short>(), &value, sizeof(value));

    CHECK(arg.get<short>() == 7);
    CHECK(arg.buffer_size() == 0);
    CHECK(arg.to_primitive() == 7);
}

TEST_CASE("KernelArg opaque type") {
    std::byte small[16] = {};
    small[3] = std::byte {42};
    TypeInfo small_type = TypeInfo::opaque("DeviceStruct", sizeof(small), 8);
    KernelArg arg = KernelArg::unsafe_construct(small_type, small, sizeof(small));

    CHECK(arg.type() == small_type);
    CHECK(arg.type().name() == "DeviceStruct");
    CHECK(static_cast<const std::byte*>(arg.data())[3] == std::byte {42});
    CHECK(arg.to_primitive_or_empty().is_empty());
    CHECK_THROWS(KernelArg::unsafe_construct(small_type, small, 8));

    // Larger than the inline storage of `KernelArg`
    std::byte large[256] = {};
    large[200] = std::byte {7};
    TypeInfo large_type = TypeInfo::opaque("DeviceStruct", sizeof(large), 16);
    KernelArg large_arg = KernelArg::unsafe_construct(large_type, large, sizeof(large));
    KernelArg copy = large_arg;

    CHECK(large_type != small_type);
    CHECK(static_cast<const std::byte*>(large_arg.data())[200] == std::byte {7});
    CHECK(static_cast<const std::byte*>(copy.data())[200] == std::byte {7});
    CHECK(copy.data() != large_arg.data());

    arg = large_arg;
    CHECK(arg.type() == large_type);
    CHECK(static_cast<const std::byte*>(arg.data())[200] == std::byte {7});
}

TEST_CASE("into_kernel_arg") {
    CHECK(into_kernel_arg(5).get<int>() == 5);
    CHECK(into_kernel_arg(2.5).get<double>() == 2.5);

    const float* ptr = nullptr;
    KernelArg arg = into_kernel_arg(ptr);
    CHECK(arg.type() == TypeInfo::of<const float*>());
    CHECK(arg.buffer_size() == 0);
}

TEST_CASE("gpu_span") {
    float data[4] = {0};
    gpu_span<float> span(data, 4);
    gpu_span<float> copy = span;
    gpu_span<const float> const_span = span;

    CHECK(copy.data() == data);
    CHECK(const_span.data() == data);
    CHECK(const_span.size() == 4);

    KernelArg arg = into_kernel_arg(const_span);
    CHECK(arg.type() == TypeInfo::of<const float*>());
    CHECK(arg.buffer_size() == 4 * sizeof(float));
}
