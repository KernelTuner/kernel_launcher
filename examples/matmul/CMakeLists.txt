cmake_minimum_required(VERSION 3.17)

set (PROJECT_NAME kernel_launcher_matmul)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/main.cu")
target_link_libraries(${PROJECT_NAME} kernel_launcher)

