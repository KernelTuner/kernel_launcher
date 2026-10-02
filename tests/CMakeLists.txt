file(GLOB_RECURSE test_sources "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(kernel_launcher_tests ${test_sources})

target_include_directories(kernel_launcher_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(kernel_launcher_tests PRIVATE "${PROJECT_SOURCE_DIR}/thirdparty")
target_link_libraries(kernel_launcher_tests PRIVATE kernel_launcher CUDA::cuda_driver)
target_compile_definitions(kernel_launcher_tests PRIVATE KERNEL_LAUNCHER_TESTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
