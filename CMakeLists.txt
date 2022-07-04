cmake_minimum_required(VERSION 3.16)
project(kernel_launcher)
set(CMAKE_BUILD_TYPE Debug)

file(GLOB sources "src/*cpp")
add_library(kernel_launcher ${sources})

target_include_directories(kernel_launcher PUBLIC include)
target_include_directories(kernel_launcher PRIVATE thirdparty)
target_compile_options(kernel_launcher PRIVATE -Wall -Wextra -pedantic -Wconversion -Werror -Wno-unused-parameter)

find_package(Threads REQUIRED)
target_link_libraries(kernel_launcher Threads::Threads)

find_package(CUDA REQUIRED)
target_include_directories(kernel_launcher PRIVATE ${CUDA_TOOLKIT_INCLUDE})
target_link_libraries(kernel_launcher ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY})
target_link_libraries(kernel_launcher nvrtc)  # Not offered by CUDA package?

add_subdirectory(tests)
