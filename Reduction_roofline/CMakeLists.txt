cmake_minimum_required(VERSION 3.22.1)
project(CUDA_OpenMP_Project LANGUAGES CXX CUDA)

# Enable OpenMP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)


find_package(OpenMP REQUIRED)

# Define the executable
include_directories(headers)
add_subdirectory(src)