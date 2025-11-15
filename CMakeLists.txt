cmake_minimum_required(VERSION 3.30)

project(DBSCAN
    VERSION 0.1.0
    DESCRIPTION "DBSCAN C++20 implementation"
    LANGUAGES CXX C
)

# ---------------------------
#  Project / options
# ---------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_SANITIZERS "Enable Address/UndefinedBehavior sanitizers (Debug builds recommended)" OFF)
option(ENABLE_EXPERIMENTAL_WARNINGS "Enable extra/experimental warnings" OFF)

# Default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Only allow GCC or Clang
if (NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "This project supports only GCC or Clang compilers. Detected: ${CMAKE_CXX_COMPILER_ID}")
else()
    message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
endif()

# Export compile_commands.json (in build dir); we will symlink it into source dir
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Symlink compile_commands.json into project root (UNIX-only)
add_custom_target(link_compile_commands ALL
    COMMENT "Symlink compile_commands.json -> ${CMAKE_SOURCE_DIR}/compile_commands.json"
)
add_custom_command(TARGET link_compile_commands POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/compile_commands.json
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json
    COMMENT "Symlinked compile_commands.json into project root"
)

function(set_strict_warnings target)
    # Common, practical warning set for GCC/Clang
    target_compile_options(${target} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Wnull-dereference
        -Wdouble-promotion
        -Wcast-align
        -Wnon-virtual-dtor
        -Woverloaded-virtual
        -Wformat=2
        -Wmissing-declarations
        -Wpointer-arith
        -Wwrite-strings
        -Werror               # treat warnings as errors
        -fno-common           # help catch ODR issues
        -fstrict-aliasing
    )

    if(ENABLE_EXPERIMENTAL_WARNINGS)
        target_compile_options(${target} PRIVATE
            -Wzero-as-null-pointer-constant
            -Wold-style-cast
        )
    endif()
endfunction()

function(set_optimizations target)
    # Debug / RelWithDebInfo / Release behaviors
    target_compile_options(${target} PRIVATE
        $<$<CONFIG:Debug>:-O0 -g3 -ggdb>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
        $<$<CONFIG:Release>:-O3 -march=native -mtune=native>
    )

    # Enable LTO for Release and RelWithDebInfo where supported
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.13")
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-flto> $<$<CONFIG:RelWithDebInfo>:-flto>)
        target_link_options(${target} PRIVATE $<$<CONFIG:Release>:-flto> $<$<CONFIG:RelWithDebInfo>:-flto>)
    endif()
endfunction()

function(enable_sanitizers_if_requested target)
    if (ENABLE_SANITIZERS)
        # Sanitizers are intended for debug/development builds. We'll attach them,
        # but note they may conflict with -O3/LTO; recommended to use Debug build type.
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -O1)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

# ---------------------------
#  Dependencies
# ---------------------------
find_package(TBB REQUIRED)

# ---------------------------
#  Library
# ---------------------------
add_library(DBSCAN STATIC src/DBSCAN.cxx)
target_include_directories(DBSCAN PUBLIC include)
target_link_libraries(DBSCAN PUBLIC TBB::tbb)
set_strict_warnings(DBSCAN)
set_optimizations(DBSCAN)
enable_sanitizers_if_requested(DBSCAN)

# ---------------------------
#  Test
# ---------------------------
add_executable(dbscan_test
    test/dbscan_test.cxx
)
target_link_libraries(dbscan_test PRIVATE DBSCAN)
target_include_directories(dbscan_test PRIVATE include)

set_strict_warnings(dbscan_test)
set_optimizations(dbscan_test)
enable_sanitizers_if_requested(dbscan_test)

# ---------------------------
#  Developer convenience targets
# ---------------------------

add_custom_target(format
    COMMENT "Run clang-format over source files (manual command not included)."
)

add_custom_target(tidy
    COMMENT "Run clang-tidy (configure manually to your needs)."
)

# ---------------------------
#  Messages
# ---------------------------
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Sanitizers enabled: ${ENABLE_SANITIZERS}")
