cmake_minimum_required(VERSION 3.17) # CUDAToolkit is available from 3.17

set (PROJECT_NAME kernel_launcher)
project(${PROJECT_NAME} LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Please specify CMAKE_BUILD_TYPE for kernel_launcher")
endif()

file(GLOB sources "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(${PROJECT_NAME} STATIC ${sources})

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/thirdparty")
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wconversion -Werror -Wno-unused-parameter)

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart_static)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::nvrtc)

if (KERNEL_LAUNCHER_BUILD_TEST)
    add_subdirectory(tests)
endif()
