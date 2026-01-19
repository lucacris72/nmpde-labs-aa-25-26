cmake_minimum_required(VERSION 3.12.0)
project(Elliptic_Full LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(Elliptic_Full src/main.cpp src/Full.cpp)
deal_ii_setup_target(Elliptic_Full)
