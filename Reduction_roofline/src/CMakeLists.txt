set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(${PROJECT_NAME} main.cpp)


target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17 cuda_std_17 )

if(OpenMP_CXX_FOUND)
    target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endif()