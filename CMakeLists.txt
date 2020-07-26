cmake_minimum_required(VERSION 3.5)
project(LKOpticalFLow)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set (CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR /home/wyl/opencv-4.2.0/cmake-build-debug)
find_package(OpenCV 4.2.0 REQUIRED)

add_executable(main main.cpp OpticalFlowSingle.cpp)

include_directories( OpticalFlowSingle.h ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS})

