cmake_minimum_required(VERSION 3.10)
project(demo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

file(GLOB SOURCES "src/*.cpp")
add_executable(demo ${SOURCES})

target_link_libraries(demo ${OpenCV_LIBS})