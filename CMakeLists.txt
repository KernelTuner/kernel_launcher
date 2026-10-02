cmake_minimum_required(VERSION 3.17) # CUDAToolkit is available from 3.17

set (PROJECT_NAME kernel_launcher)
project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(KERNEL_LAUNCHER_TOP_LEVEL ON)
endif()

option(KERNEL_LAUNCHER_BUILD_TEST "Build the kernel_launcher tests" OFF)
option(KERNEL_LAUNCHER_BUILD_EXAMPLE "Build the kernel_launcher examples" OFF)
option(KERNEL_LAUNCHER_ZLIB "Enable zlib support in kernel_launcher" OFF)
option(KERNEL_LAUNCHER_EMBEDDED_DATA "Enable embedded data in kernel_launcher" ON)

# Enable clang-tidy by default only if this is the top-level project
if (KERNEL_LAUNCHER_TOP_LEVEL)
    set(KERNEL_LAUNCHER_CLANG_TIDY_DEFAULT clang-tidy -checks=-*,readability-*,bugprone-*,-readability-magic-numbers,-readability-use-anyofallof,-readability-else-after-return,-readability-identifier-length,-bugprone-easily-swappable-parameters,-bugprone-assignment-in-if-condition)
endif()
set(KERNEL_LAUNCHER_CLANG_TIDY "${KERNEL_LAUNCHER_CLANG_TIDY_DEFAULT}" CACHE STRING "clang-tidy command for kernel_launcher (empty to disable)")

get_property(KERNEL_LAUNCHER_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(KERNEL_LAUNCHER_TOP_LEVEL AND NOT KERNEL_LAUNCHER_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Please specify CMAKE_BUILD_TYPE for kernel_launcher")
endif()


file(GLOB sources "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*/*.cpp")
add_library(${PROJECT_NAME} STATIC ${sources})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/thirdparty")
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wconversion -Wno-unused-parameter)

if (KERNEL_LAUNCHER_TOP_LEVEL)
    target_compile_options(${PROJECT_NAME} PRIVATE -Werror)
endif()

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart_static)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cuda_driver)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::nvrtc)


##
# Embed files from a directory into a target.
#
# Parameters:
#   target: The target for which the embedded files are created
#   directory: The directory where the files are located
#   files...: Multiple files can be listed
#
# Usage:
#   kernel_launcher_embed(my_target my_directory file1.bin file2.bin)
##
function(kernel_launcher_embed target directory)
    string(MAKE_C_IDENTIFIER ${target}_embed_${directory} output_target)
    set(deps "")

    foreach(input IN LISTS ARGN)
        string(MAKE_C_IDENTIFIER ${input} input_identifier)
        set(output "${CMAKE_CURRENT_BINARY_DIR}/${input_identifier}.o")

        add_custom_command(
            OUTPUT ${output}
            COMMAND ${CMAKE_LINKER} --relocatable --format binary --output ${output} ${input}
            WORKING_DIRECTORY ${directory}
            DEPENDS "${directory}/${input}"
        )

        set(deps ${deps} "${output}")
    endforeach()

    add_custom_target(${output_target} ALL DEPENDS ${deps})

    add_dependencies(${target} ${output_target})
    target_link_libraries(${target} PRIVATE ${output})
endfunction()

##
# Embed files from a directory into a target with the ability to specify the embedded files using a glob pattern.
#
# Parameters:
#   target: The target for which the embedded files are created
#   directory: The directory where the files are located
#   patterns...: Multiple patterns can be listed
#
# Usage:
#   kernel_launcher_embed_glob(my_target my_directory *.cuh *.h)
##
function(kernel_launcher_embed_glob target directory)
    list(TRANSFORM ARGN PREPEND "${directory}/" OUTPUT_VARIABLE patterns)
    file(GLOB_RECURSE files LIST_DIRECTORIES false RELATIVE "${directory}" ${patterns})
    kernel_launcher_embed("${target}" "${directory}" ${files})
endfunction()

##
# Embed all files from a directory into a target.
#
# Parameters:
#   target: The target for which the embedded files are created
#   directory: The directory where the files are located
#
# Usage:
#   kernel_launcher_embed_directory(my_target my_directory)
##
function(kernel_launcher_embed_directory target directory)
    kernel_launcher_embed_glob("${target}" "${directory}" "*")
endfunction()

if (KERNEL_LAUNCHER_CLANG_TIDY)
    set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "${KERNEL_LAUNCHER_CLANG_TIDY}")
endif()

if (KERNEL_LAUNCHER_EMBEDDED_DATA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE KERNEL_LAUNCHER_EMBEDDED_DATA=1)
endif()

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

