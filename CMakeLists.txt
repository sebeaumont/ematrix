project(analysis)

cmake_minimum_required (VERSION 3.0)

# extra cmake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


#########################################
# project global c/c++ compiler settings
#########################################

# common to both c & c++ flags
set(MY_ALL_FLAGS "-Wall -O3 -march=native")

# c++ only
set(MY_CXX_FLAGS "-std=c++14")

# c only
set(MY_C_FLAGS "-std=c11")


# is future namespace wizardy allowed?
# YES: set(MY_CXX_FLAGS "-Wc++1z-extensions")


############################################
# compiler specific project wide defintions


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # TODO for clang these are clang specific
  # set(MY_CXX_FLAGS "-Rpass=.*") # have clang show optimizations
  # set(MY_CXX_FLAGS "-fsanitize=address") doesn't work with apple 3.6 compiler!
  set(MY_ALL_FLAGS "${MY_ALL_FLAGS} -funroll-loops -fvectorize -fslp-vectorize -fslp-vectorize-aggressive")
endif()


######################################################################
# we use think different threading backends on Darwin derived systems
# also we probe what cpu features we can exploit...

set(CPU_FEATURES)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # OS X/XNU specific configuration - should work for iOS?
  message(STATUS "Will compile for Blocks/GCD and other Apple sweetness...")
  set(HAVE_DISPATCH 1)
  set(MY_CXX_FLAGS "${MY_CXX_FLAGS} -stdlib=libc++")
  set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-stdlib=libc++")
  exec_program("sysctl -n machdep.cpu.features" OUTPUT_VARIABLE CPU_FEATURES)

else()

  # Use openmp on non OSX Systems (linux)
  
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)

    message(STATUS "OpenMP is configured")
    set(HAVE_OPENMP 1)
    set(MY_ALL_FLAGS "${MY_ALL_FLAGS} -fopenmp")
    set(MY_C_FLAGS "${MY_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(MY_CXX_FLAGS "${MY_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS})
    
    find_package(Threads REQUIRED)

    if (CMAKE_USE_PTHREADS_INIT)
      set(MY_CXX_FLAGS "${MY_CXX_FLAGS} -pthread")
    endif()

    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_THREAD_LIBS_INIT})
    
    file(READ "/proc/cpuinfo" _cpuinfo)
    string(REGEX REPLACE ".*flags[ \t]*:[ \t]+([a-zA-Z0-9_ ]+).*" "\\1" CPU_FEATURES "${_cpuinfo}")
  endif()

endif()

# Now set the real CXX and C FLAGS
set(CMAKE_CXX_FLAGS "${MY_CXX_FLAGS} ${MY_ALL_FLAGS}")
set(CMAKE_C_FLAGS "${MY_C_FLAGS} ${MY_ALL_FLAGS}")

# let's see
message(${CMAKE_CXX_FLAGS})
message(${CMAKE_C_FLAGS})

# normalise features before searching
string(TOUPPER ${CPU_FEATURES} CPU_FEATURES)

#######################
# now the actual build 
#######################

include_directories(.)

# boost
find_package(Boost REQUIRED COMPONENTS program_options)
include_directories (${Boost_INCLUDE_DIR})

# eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#####################
# command line tools

add_executable (feature_space feature_space.cpp)
target_link_libraries(feature_space ${Boost_LIBRARIES})

# test io
add_executable (testio testio.cpp)







