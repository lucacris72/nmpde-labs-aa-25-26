cmake_minimum_required(VERSION 3.12.0)
project(Parabolic_Full LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(Parabolic_Full src/main.cpp src/Parabolic.cpp)
deal_ii_setup_target(Parabolic_Full)
