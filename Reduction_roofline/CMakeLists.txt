cmake_minimum_required(VERSION 3.21)


project(
	roofline_reduction 
	LANGUAGES CXX CUDA
	)
	
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 80 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


include_directories(headers)
add_subdirectory(src)