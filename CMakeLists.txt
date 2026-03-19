cmake_minimum_required(VERSION 3.10)
project(demo)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/opencv4")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

file(GLOB SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/demo.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/demo_cluster.cpp")

add_executable(demo src/demo.cpp ${SOURCES})
target_link_libraries(demo ${OpenCV_LIBS})

add_executable(demo_cluster src/demo_cluster.cpp ${SOURCES})
target_link_libraries(demo_cluster ${OpenCV_LIBS})