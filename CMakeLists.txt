cmake_minimum_required(VERSION 3.16)

set (PROJECT_NAME kernel_launcher)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Please specify CMAKE_BUILD_TYPE for kernel_launcher")
endif()

file(GLOB sources "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(${PROJECT_NAME} ${sources})

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/thirdparty")
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wconversion -Werror -Wno-unused-parameter)

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads)

find_package(CUDA REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_TOOLKIT_INCLUDE})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} cuda)
target_link_libraries(${PROJECT_NAME} nvrtc)  # Not offered by CUDA package?

if (KERNEL_LAUNCHER_BUILD_TEST)
    add_subdirectory(tests)
endif()
