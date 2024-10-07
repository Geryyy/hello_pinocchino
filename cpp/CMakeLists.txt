cmake_minimum_required(VERSION 3.10)
project(RobocraneSimulator)

# Set PKG_CONFIG_PATH to include /usr/local/lib/pkgconfig
set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")


# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find Pinocchio (assuming it's installed via pkg-config)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PINOCCHIO REQUIRED pinocchio>=3.2.0)


# Find hpp-fcl (assuming it's installed via pkg-config)
pkg_check_modules(HPP_FCL REQUIRED hpp-fcl)

# Add include directories
include_directories(
    ${PINOCCHIO_INCLUDE_DIRS}
    ${HPP_FCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Add the executable
add_executable(robocrane_sim main.cpp)

# Link the libraries
target_link_libraries(robocrane_sim 
    ${PINOCCHIO_LIBRARIES}
    ${HPP_FCL_LIBRARIES}
    Eigen3::Eigen
    dl  # For dynamic loading (if needed)
    pthread  # Multi-threading
)
