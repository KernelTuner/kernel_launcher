cmake_minimum_required(VERSION 3.17) # CUDAToolkit is available from 3.17

set (PROJECT_NAME kernel_launcher)
project(${PROJECT_NAME} LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Please specify CMAKE_BUILD_TYPE for kernel_launcher")
endif()


file(GLOB sources "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*/*.cpp")
add_library(${PROJECT_NAME} STATIC ${sources})
set(KERNEL_LAUNCHER_CLANG_TIDY clang-tidy -checks=-*,readability-*,bugprone-*,-readability-magic-numbers,-readability-use-anyofallof,-readability-else-after-return)

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/thirdparty")
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wconversion -Werror -Wno-unused-parameter)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "${KERNEL_LAUNCHER_CLANG_TIDY}")

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart_static)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::nvrtc)

if (KERNEL_LAUNCHER_BUILD_TEST)
    add_subdirectory(tests)
endif()

if (KERNEL_LAUNCHER_BUILD_EXAMPLE)
    add_subdirectory(examples)
endif()

if (KERNEL_LAUNCHER_ZLIB)
    find_package(ZLIB REQUIRED)
    target_include_directories(${PROJECT_NAME} PRIVATE ${ZLIB_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ZLIB_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE KERNEL_LAUNCHER_USE_ZLIB=1)
endif()
