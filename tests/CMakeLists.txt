cmake_minimum_required(VERSION 3.16)

project(tests)
set(CMAKE_CXX_STANDARD 17)

file(GLOB FILES *.cpp)
add_executable(kernel_launcher_tests ${FILES})
target_link_libraries(kernel_launcher_tests PRIVATE kernel_launcher)

find_package(CUDA REQUIRED)
target_include_directories(kernel_launcher_tests PRIVATE ${CUDA_TOOLKIT_INCLUDE})

target_include_directories(kernel_launcher_tests PRIVATE "${PROJECT_SOURCE_DIR}/../thirdparty")
