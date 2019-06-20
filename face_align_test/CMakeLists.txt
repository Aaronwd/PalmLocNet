cmake_minimum_required(VERSION 2.8)
project(face_alignment)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_BUILD_TYPE Release)

#add opencv
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(OTHER_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/include/libface_align_library.so)

add_executable(test_face_align main.cpp)
target_link_libraries(test_face_align ${OpenCV_LIBS} ${OTHER_LIBS})

