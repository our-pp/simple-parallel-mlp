cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_ver)

find_package(Torch REQUIRED)

add_executable(torch_ver torch_ver.cpp)
target_link_libraries(torch_ver "${TORCH_LIBRARIES}")
set_property(TARGET torch_ver PROPERTY CXX_STANDARD 17)
