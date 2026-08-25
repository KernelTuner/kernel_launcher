cmake_minimum_required(VERSION 3.17)

set (PROJECT_NAME kernel_launcher_convolution)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/main.cu")
target_link_libraries(${PROJECT_NAME} kernel_launcher)

# Optional: bake the .cu source into the binary so it does not need to be
# found on disk at runtime.
# kernel_launcher_embed_glob(convolution_example ${PROJECT_SOURCE_DIR} *.cu *.cuh)
