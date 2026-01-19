cmake_minimum_required(VERSION 3.12.0)
project(Stokes_Full LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(Stokes_Full src/main.cpp src/Stokes.cpp)
deal_ii_setup_target(Stokes_Full)