cmake_minimum_required(VERSION 3.10)
project(simple_camera)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(simple_camera simple_camera.cpp)
target_link_libraries(simple_camera ${OpenCV_LIBS})