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
            DEPENDS ${input}
        )

        set(deps ${deps} "${output}")
    endforeach()

    add_custom_target(${output_target} ALL DEPENDS ${deps})

    add_dependencies(${target} ${output_target})
    target_link_libraries(${target} ${output})
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
    file(GLOB_RECURSE files LIST_DIRECTORIES false RELATIVE "${directory}" ${ARGN})
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
    kernel_launcher_embed_glob(target directory "*")
endfunction()

if (NOT DEFINED KERNEL_LAUNCHER_EMBEDDED_DATA)
    set(KERNEL_LAUNCHER_EMBEDDED_DATA 1)
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

