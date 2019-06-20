cmake_minimum_required(VERSION 2.8)
project(face_alignment)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_BUILD_TYPE Release)

#add opencv
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(face_align_library SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/face_align.cpp)

add_executable(face_align main.cpp)
target_link_libraries(face_align ${OpenCV_LIBS} face_align_library)

