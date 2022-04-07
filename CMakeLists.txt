cmake_minimum_required(VERSION 3.8)

# set(CMAKE_BUILD_TYPE "Debug")
IF (WIN32)
    message(STATUS "In windows")
    SET(CMAKE_MAKE_PROGRAM "D:/Program Files/MGW/mingw64/bin/make.exe")

    SET(CMAKE_C_COMPILER "D:/Program Files/MGW/mingw64/bin/gcc.exe")
    SET(CMAKE_CXX_COMPILER "D:/Program Files/MGW/mingw64/bin/g++.exe")
ENDIF()

project(mycv)
add_subdirectory(pybind11)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
pybind11_add_module(mycv src/mycv.cpp)