cmake_minimum_required(VERSION 3.24.0)
project(Test)

add_executable(${PROJECT_NAME} test_main.cpp)

find_package(mylib)
target_link_libraries(${PROJECT_NAME} PRIVATE mylib)