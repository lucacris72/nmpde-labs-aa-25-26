cmake_minimum_required(VERSION 3.12.0)
project(Stokes_Time_Full LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(Stokes_Time_Full src/main.cpp src/StokesTime.cpp)
deal_ii_setup_target(Stokes_Time_Full)