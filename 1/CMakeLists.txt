cmake_minimum_required(VERSION 3.28)
project(opencv_tasks)

set(CMAKE_CXX_STANDARD 17)

file(COPY samples DESTINATION .)

include("~/dev/lib/opencv/build/OpenCVConfig.cmake")
add_executable(opencv_tasks main.cpp)
target_include_directories(opencv_tasks PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_tasks ${OpenCV_LIBS})

#find_package(OpenCV REQUIRED PATHS "~/dev/lib/opencv/build")
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(opencv_tasks $ {OpenCV_LIBS})
