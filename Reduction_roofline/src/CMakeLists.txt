add_executable(${PROJECT_NAME} main.cu)


target_compile_options(${PROJECT_NAME}  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>) 
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17 cuda_std_17 )

if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
    target_link_libraries(${PROJECT_NAME} PRIVATE  OpenMP::OpenMP_CXX)
    target_compile_options(${PROJECT_NAME} PRIVATE  ${OpenMP_CXX_FLAGS})
endif()
