cmake_minimum_required(VERSION "3.19.2")


project("gekko_ml")
set(CMAKE_CXX_STANDARD 17)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
add_executable(main "src/main.cpp")
target_link_libraries(main xtensor xtensor::optimize)
target_include_directories(main PUBLIC "include/")

