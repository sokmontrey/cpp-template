cmake_minimum_required(VERSION 3.24)

# Configuration of the project needs to set VERSION to generate .cmake files and
# to use find_package.
project(
    mylib
    LANGUAGES CXX
    VERSION 0.0.2
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# Set default library path to install
include(GNUInstallDirs)
# Features to generate .cmake file for find_package.
include(CMakePackageConfigHelpers)

# Build of the library
# add_subdirectory(source)
add_library(${PROJECT_NAME}
    # Header and Sources Files --------------------------------------
    source/mylib/mylib.h
    source/mylib/mylib.cpp
    # END Header and Source Files -----------------------------------
)
# add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/source>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# External Library --------------------------------------------------
# find_package(extern_lib REQUIRED)
# target_link_libraries(${PROJECT_NAME} extern_lib)

# END External Library ----------------------------------------------

# Test --------------------------------------------------------------
option(BUILD_TESTING "Build tests" ON)
include(CTest)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
# END Test ----------------------------------------------------------

# Enable tests.
# BUILD_TESTING option is defined via include(CTest)

# build
export(TARGETS ${PROJECT_NAME} 
       NAMESPACE ${PROJECT_NAME}::
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

# package
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# install
install(
    TARGETS ${PROJECT_NAME} 
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# install header
install(
    DIRECTORY source/${PROJECT_NAME}  # Adjust the path to your header files
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)