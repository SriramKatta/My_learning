cmake_minimum_required(VERSION 3.18)
project(MyCUDAProject LANGUAGES CUDA CXX)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)



# Add the executable
add_executable(my_cuda_executable main.cpp)

# Specify that main.cpp should be compiled as a CUDA file
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

# Enable CUDA separable compilation (optional but recommended)
set_target_properties(my_cuda_executable PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
