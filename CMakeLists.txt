cmake_minimum_required(VERSION 3.10)
project(demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")

set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/opencv4")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

file(GLOB SOURCES "src/*.cpp")
file(GLOB DEMO_SOURCES "src/demo_*.cpp")
list(REMOVE_ITEM SOURCES ${DEMO_SOURCES})

foreach(DEMO_CPP IN LISTS DEMO_SOURCES)
  get_filename_component(EXE_NAME ${DEMO_CPP} NAME_WE)
  add_executable(${EXE_NAME} ${DEMO_CPP} ${SOURCES})
  target_link_libraries(${EXE_NAME} ${OpenCV_LIBS})
endforeach()