cmake_minimum_required(VERSION "3.19.2")


project("ray_tracing")
set(CMAKE_CXX_STANDARD 17)
add_executable(main "src/neuron.cpp")
target_include_directories(main PUBLIC "include/")