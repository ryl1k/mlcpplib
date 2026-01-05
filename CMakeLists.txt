cmake_minimum_required(VERSION 3.16)
project(mlcpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mlcpp
  src/core/shape.cpp
  src/tensor/tensor.cpp
 "src/ops/matmul.cpp" "src/ops/elementwise.cpp" "tests/test_scalar_autograd.cpp")

target_include_directories(mlcpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_tensor tests/test_tensor.cpp "src/ops/matmul.cpp" "src/ops/elementwise.cpp" "tests/test_scalar_autograd.cpp")
target_link_libraries(test_tensor PRIVATE mlcpp)
