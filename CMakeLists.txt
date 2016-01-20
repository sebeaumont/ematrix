project(analysis)

cmake_minimum_required (VERSION 3.0)

# extra cmake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories(.)

# boost
#find_package(Boost REQUIRED COMPONENTS program_options filesystem system)
find_package(Boost REQUIRED COMPONENTS program_options)
include_directories (${Boost_INCLUDE_DIR})

# eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


# modern c++ please
add_definitions("-std=c++11")

# YMMV so this is where we tune for our target/platform
add_definitions("-O3")
add_definitions("-mavx")
add_definitions("-funroll-loops")
add_definitions("-fvectorize")
add_definitions("-fslp-vectorize")
add_definitions("-fslp-vectorize-aggressive")

# todo library?
# add_library(fspace feature_space.cpp)

# command line tool
add_executable (feature_space feature_space.cpp)
target_link_libraries(feature_space ${Boost_LIBRARIES})






