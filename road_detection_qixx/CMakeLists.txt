CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(RoadDetection)

# 查找opencv
set(OpenCV_DIR /usr/local/include/opencv) 
FIND_PACKAGE(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 头文件目录

# 执行程序
ADD_EXECUTABLE(RoadDetection main.cpp)

target_link_libraries(RoadDetection ${OpenCV_LIBS})



