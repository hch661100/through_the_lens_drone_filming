cmake_minimum_required(VERSION 2.8.3)
project(through_the_lens_drone_filming)

## Add support for C++11, supported in ROS Kinetic and newer
set(CMAKE_VERBOSE_MAKEFILE "true")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -Wall")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  image_recognition_msgs
  cv_bridge
  image_transport
  gflags
  glog_catkin
  message_filters
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
# find_package(CUDA REQUIRED) # Uncomment this line if you have trouble compiling
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${catkin_INCLUDE_DIRS} )

set(CAFFE_DIR /home/nvidia/openpose/3rdparty/caffe)
set(OPENPOSE_DIR /home/nvidia/openpose)

set(CAFFE_INCLUDE_DIRS ${CAFFE_DIR}/distribute/include)
set(CAFFE_LINK_LIBRARY ${CAFFE_DIR}/distribute/lib/libcaffe.so)

set(OPENPOSE_INCLUDE_DIRS ${OPENPOSE_DIR}/include)
set(OPENPOSE_LINK_LIBRARY ${OPENPOSE_DIR}/build/lib/libopenpose.so)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES openpose_ros
   CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories( include )
#include_directories(
  #/usr/local/cuda-8.0/targets/aarch64-linux/include
  #/home/nvidia/openpose/include
  #${catkin_INCLUDE_DIRS}
  #${CUDA_INCLUDE_DIRS}
  #${CAFFE_INCLUDE_DIRS}
  #${OPENPOSE_INCLUDE_DIRS}
#)
link_directories( 
  /usr/local/lib 
  /usr/lib 
  /usr/lib/aarch64-linux-gnu
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  /usr/local/cuda-8.0/targets/aarch64-linux/include
  /home/nvidia/openpose/include
  /home/nvidia/openpose/3rdparty/caffe/include
  /home/nvidia/openpose/3rdparty/caffe/.build_release/src
)


add_executable(extra_2d_pose_node src/extra_2d_pose.cpp)
add_dependencies(extra_2d_pose_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(extra_2d_pose_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  #${CAFFE_LINK_LIBRARY}
  #${OPENPOSE_LINK_LIBRARY}
  libcudnn.so
  /home/nvidia/openpose/build/lib/libopenpose.so
  /home/nvidia/openpose/3rdparty/caffe/.build_release/lib/libcaffe.so
)

add_executable(control_cam src/control_cam.cpp)
add_dependencies(control_cam ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(control_cam
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

