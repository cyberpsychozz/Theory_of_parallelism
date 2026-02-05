cmake_minimum_required(VERSION 3.10)
project(SineSum)

option(USE_DOUBLE "Use double precision (default: float)" OFF)

if(USE_DOUBLE)
    add_definitions(-DUSE_DOUBLE)
endif()

add_executable(sinesum main.cpp)
target_compile_features(sinesum PRIVATE cxx_std_11)