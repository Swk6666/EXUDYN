cmake_minimum_required(VERSION 3.16)
project(flexible_pendulum_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT Python3_ROOT_DIR)
    message(STATUS "Python3_ROOT_DIR not provided; falling back to system detection")
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(PYBIND11_INCLUDE_HINT
    "${CMAKE_CURRENT_LIST_DIR}/../../../include/pybind11local"
)
if(NOT EXISTS "${PYBIND11_INCLUDE_HINT}/pybind11/pybind11.h")
    message(FATAL_ERROR "pybind11 headers not found at ${PYBIND11_INCLUDE_HINT}; adjust PYBIND11_INCLUDE_DIR")
endif()

add_executable(flexible_pendulum_cpp flexible_pendulum_cpp.cpp)

target_include_directories(flexible_pendulum_cpp
    PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_HINT}
)

target_link_libraries(flexible_pendulum_cpp PRIVATE Python3::Python)

set_target_properties(flexible_pendulum_cpp PROPERTIES
    INSTALL_RPATH "${Python3_RUNTIME_LIBRARY_DIRS}"
    BUILD_RPATH   "${Python3_RUNTIME_LIBRARY_DIRS}"
)

message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python libs: ${Python3_LIBRARIES}")
