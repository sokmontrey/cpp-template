# This Sub Library -------------------------------------------------------------------
set(LIB_NAME "mylib")

add_library(${LIB_NAME} STATIC
    mylib.h
    mylib.cpp
)

set(TARGETS ${LIB_NAME})

add_library(${PROJECT_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})

target_include_directories(${LIB_NAME}
    PUBLIC $<BUILD_INTERFACE:${THIS_LIB_DIR}/source>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# set_target_properties(${LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Shared Libraries export ----------------------------------------------------------

include(GenerateExportHeader)
generate_export_header(${LIB_NAME}
        BASE_NAME ${LIB_NAME}
        EXPORT_MACRO_NAME MATPLOT_EXPORTS
        EXPORT_FILE_NAME ${LIB_NAME}/detail/exports.h
        STATIC_DEFINE SHARED_EXPORTS_BUILT_AS_STATIC)
target_include_directories(${LIB_NAME}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}/detail/exports.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}/detail)

# Dependencies ------------------------------------------------------------------------
include(../../cmake/functions.cmake)
# target_link_libraries_system(${LIB_NAME} PRIVATE lib_name)

# Required compiler features required
# https://cmake.org/cmake/help/v3.14/manual/cmake-compile-features.7.html#requiring-language-standards
target_compile_features(${LIB_NAME} PUBLIC cxx_std_17)

# Installer ---------------------------------------------------------------------------

if (BUILD_INSTALLER)
    # Install targets
    install(TARGETS ${TARGETS}
            EXPORT ${PROJECT_NAME}Targets
	    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            )

    # Install headers
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h"
            )

    # Install cmake script
    install(EXPORT ${PROJECT_NAME}Targets
            FILE ${PROJECT_NAME}Targets.cmake
            NAMESPACE ${PROJECT_NAME}::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
            )
endif()