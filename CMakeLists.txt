cmake_minimum_required(VERSION 3.24)
project(ParallelProgrammingCUDA CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(ParallelProgrammingCUDA main.cu)

set_target_properties(ParallelProgrammingCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
