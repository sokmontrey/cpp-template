cmake_minimum_required(VERSION 3.24.0)
project(Test)

add_executable(test_main test_main.cpp)
target_link_libraries(test_main mylib)