cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(fast_mmio LANGUAGES C CXX)

#add_subdirectory(header)

add_executable(${PROJECT_NAME} main.cpp)
#target_link_libraries(${PROJECT_NAME} PRIVATE mmio)

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)