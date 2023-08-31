cmake_minimum_required(VERSION 3.24.0)

project(mylib VERSION 0.0.1)

include(cmake/functions.cmake)

set(THIS_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Check if project is Master or not ----------------------------------------------

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(MASTER_PROJECT ON)
else ()
    set(MASTER_PROJECT OFF)
endif ()

# Set some booleans according to ${MASTER_PROJECT}

if(${MASTER_PROJECT})
    option(BUILD_TESTING "Build Tests" ON)
else()
    option(BUILD_TESTING "Build Tests" OFF)
endif()

option(BUILD_INSTALLER "Build Installer Target" ${MASTER_PROJECT})
option(BUILD_PACKAGE "Build Package" ${MASTER_PROJECT})

# Library ------------------------------------------------------------------------

add_subdirectory(source)

# Test ---------------------------------------------------------------------------

if(BUILD_TESTING)
    include(CTEST)
    enable_testing()
    add_subdirectory(test)
endif()

# Installer ----------------------------------------------------------------------

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if (BUILD_INSTALLER)
    # Set variable where the cmake config is
    set(CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
    message("CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
    message("CMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}")

    # Create ${PROJECT_NAME}ConfigVersion.cmake and install it
    write_basic_package_version_file(
            ${PROJECT_NAME}ConfigVersion.cmake
            VERSION ${PACKAGE_VERSION}
            COMPATIBILITY SameMajorVersion
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

    # Create ${PROJECT_NAME}Config.cmake from ${PROJECT_NAME}Config.cmake.in
    set(INCLUDE_INSTALL_DIR include/)
    set(LIB_INSTALL_DIR lib/)
    configure_package_config_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in # input file
            ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake    # output file
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
            PATH_VARS CMAKE_INSTALL_LIBDIR INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
endif ()

# Packaging ----------------------------------------------------------------------------

# if (BUILD_INSTALLER AND BUILD_PACKAGE)
#     # The most common cpack variables
#     set(CPACK_PACKAGE_VENDOR ${PROJECT_NAME})
#     set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_NAME}: Description")
#     set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
#     set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
#     set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
#     set(CPACK_RESOURCE_FILE_LICENSE "${THIS_LIB_DIR}/LICENSE")
#     set(CPACK_RESOURCE_FILE_README "${THIS_LIB_DIR}/README.md")

#     # Set CPACK_SOURCE_IGNORE_FILES with files source packages shouldn't install
#     # We get these from .gitignore to avoid redundancy
#     FILE(READ .gitignore GITIGNORE_CONTENTS)
#     STRING(REGEX REPLACE ";" "\\\\;" GITIGNORE_CONTENTS "${GITIGNORE_CONTENTS}")
#     STRING(REGEX REPLACE "\n" ";" GITIGNORE_CONTENTS "${GITIGNORE_CONTENTS}")
#     set(CPACK_SOURCE_IGNORE_FILES ${GITIGNORE_CONTENTS})

#     # Always include CPack at last
#     include(CPack)
# endif ()