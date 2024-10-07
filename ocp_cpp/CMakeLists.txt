cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_COMPILER "gcc-10")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")


project(ocp_cpp)

# Set PKG_CONFIG_PATH to include /usr/local/lib/pkgconfig
set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")


# ignore Anaconda compilers, which are typically not compatible with the system
set(CMAKE_IGNORE_PATH $ENV{CONDA_PREFIX})
message("CMAKE_IGNORE_PATH: ${CMAKE_IGNORE_PATH}")

# Output the compiler type and version
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

# exclude Anaconda directories from search
#if(DEFINED ENV{CONDA_PREFIX})
#    list(APPEND CMAKE_IGNORE_PREFIX_PATH $ENV{CONDA_PREFIX})
#    list(APPEND CMAKE_IGNORE_PATH $ENV{CONDA_PREFIX}/bin)
#    # need CMAKE_IGNORE_PATH to ensure system env var PATH
#    # doesn't interfere despite CMAKE_IGNORE_PREFIX_PATH
#endif()

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define a custom option to enable debug settings
option(ENABLE_DEBUG "Enable debug settings" OFF)

if (ENABLE_DEBUG)
    # Debug settings
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    message("--- debug build ---")
else ()
    # Maximum performance settings
    set(CMAKE_BUILD_TYPE Release)
    #set(CMAKE_BUILD_TYPE RelWithDebInfo)
#    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    message("-- optimized build --")
endif ()

# find_package(pinocchio REQUIRED)

# Find Pinocchio (assuming it's installed via pkg-config)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PINOCCHIO REQUIRED pinocchio>=3.2.0)

find_package(Eigen3 3.4 NO_MODULE)

# Use pkg-config to find and include CppAD
find_package(PkgConfig REQUIRED)
pkg_check_modules(CPPAD REQUIRED cppad)
#pkg_check_modules(CPPADCG REQUIRED cppadcg)

include_directories(${PINOCCHIO_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${CPPAD_INCLUDE_DIRS})
#include_directories(${CPPADCG_INCLUDE_DIRS})

add_executable(codegen main_cg.cpp Timer.h)
target_link_libraries(codegen PUBLIC ${CMAKE_DL_LIBS} ${PINOCCHIO_LIBRARIES} ${CPPAD_LIBRARIES} ${CPPADCG_LIBRARIES} Eigen3::Eigen)
target_compile_definitions(codegen PUBLIC PINOCCHIO_CXX_COMPILER=\"${CMAKE_CXX_COMPILER}\")

add_executable(benchmark main_bench.cpp Timer.h)
target_link_libraries(benchmark Eigen3::Eigen ${PINOCCHIO_LIBRARIES})

add_executable(ad_fwdkin main_ad.cpp Timer.h)
target_link_libraries(ad_fwdkin Eigen3::Eigen ${PINOCCHIO_LIBRARIES} ${CPPAD_LIBRARIES})

add_executable(ocp main_ocp.cpp Timer.h
        rhtp/RHMPC.cpp
        rhtp/RHMPC.h
        rhtp/OCP.cpp
        rhtp/OCP.h)
target_link_libraries(ocp Eigen3::Eigen ${PINOCCHIO_LIBRARIES} ${CPPAD_LIBRARIES})
