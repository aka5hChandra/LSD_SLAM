cmake_minimum_required(VERSION 2.8.7)
project(lsd_slam_core)

# SSE flags
#rosbuild_check_for_sse()
add_definitions("-DUSE_ROS")
add_definitions("-DENABLE_SSE")

# Also add some useful compiler flag
set(CMAKE_CXX_FLAGS
   "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native  -std=c++0x"
) 
#set(ROS_BUILD_TYPE Release)
LIST(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(catkin REQUIRED
  roscpp
  gencpp
  rospy
  genpy
  rosbag # for main_bagfile_driver
  cmake_modules
  cv_bridge
  eigen_conversions
  geometry_msgs
  sensor_msgs
  image_geometry
  image_transport
  nodelet
  std_msgs
  tf
  tf2
  tf2_ros
  cmake_modules
  lsd_slam_viewer
  dynamic_reconfigure)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(sophus REQUIRED)

find_package(cmake_modules REQUIRED)   
find_package(Eigen3 REQUIRED)
find_package(SuiteParse REQUIRED) # Apparently needed by g2o
find_package(X11 REQUIRED)
find_package(G2O REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Dynamic Reconfigure Services
generate_dynamic_reconfigure_options(cfg/LSDDebugParams.cfg cfg/LSDParams.cfg)

if(G2O_FOUND)
  message(STATUS "Found G2O")
else()
  message(ERROR "G2O missing")
endif()

# FabMap
# uncomment this part to enable fabmap
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/openFabMap)
include_directories(
      /opt/ros/kinetic/include
    ${PROJECT_SOURCE_DIR}/thirdparty/openFabMap/include
  ${OPENCV_INCLUDE_DIR}
)
add_definitions("-DHAVE_FABMAP")
set(FABMAP_LIB openFABMAP )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rgbd_drivers_uncc
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
  DEPENDS Boost OpenCV 
  LIBRARIES ${PROJECT_NAME}
)

# Set source files
set(lsd_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/DataStructures/Frame.cpp
  ${PROJECT_SOURCE_DIR}/src/DataStructures/FramePoseStruct.cpp
  ${PROJECT_SOURCE_DIR}/src/DataStructures/FrameMemory.cpp
  ${PROJECT_SOURCE_DIR}/src/SlamSystem.cpp
  ${PROJECT_SOURCE_DIR}/src/LiveSLAMWrapper.cpp
  ${PROJECT_SOURCE_DIR}/src/DepthEstimation/DepthMap.cpp
  ${PROJECT_SOURCE_DIR}/src/DepthEstimation/DepthMapPixelHypothesis.cpp
  ${PROJECT_SOURCE_DIR}/src/util/globalFuncs.cpp
  ${PROJECT_SOURCE_DIR}/src/util/SophusUtil.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Undistorter.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/Sim3Tracker.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/Relocalizer.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/SE3Tracker.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/TrackingReference.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Timestamp.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/FabMap.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/KeyFrameGraph.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/g2oTypeSim3Sophus.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/TrackableKeyFrameSearch.cpp
)
set(SOURCE_FILES
  ${lsd_SOURCE_FILES}
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ROS/ROSImageStreamThread.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ROS/ROSOutput3DWrapper.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
)

include_directories(AFTER
  ${PROJECT_SOURCE_DIR}/cfg/cpp
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${G2O_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${OPENCV_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIR}
  ${CSPARSE_INCLUDE_DIR} #Has been set by SuiteParse
  ${CHOLMOD_INCLUDE_DIR} #Has been set by SuiteParse
)


# build shared library.
add_library(lsd_slam_core 
  SHARED 
  ${SOURCE_FILES})
add_dependencies(lsd_slam_core ${PROJECT_NAME}_gencfg)
target_link_libraries(lsd_slam_core 
   ${FABMAP_LIB} 
   ${catkin_LIBRARIES} 
   ${Boost_LIBRARIES}
   ${OpenCV_LIBRARIES}  
   ${Eigen3_LIBRARIES}  
   ${OPENGL_LIBRARIES}  
   ${GLUT_LIBRARIES}  
   ${G2O_CORE_LIBRARY}
   ${G2O_SOLVER_DENSE}
   ${G2O_SOLVER_EIGEN}
   ${G2O_TYPES_SLAM3D}
   ${G2O_TYPES_SBA}
   ${G2O_SOLVER_CSPARSE}
   ${G2O_SOLVER_CSPARSE_EXTENSION}
   csparse cxsparse X11)

# build live ros node
add_executable(live_slam 
  src/main_live_odometry.cpp)
add_dependencies(live_slam ${PROJECT_NAME}_gencfg)
target_link_libraries(live_slam 
  ${catkin_LIBRARIES} 
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}  
  ${Eigen3_LIBRARIES}  
  lsd_slam_core)

# build image node
add_executable(dataset_slam src/main_on_images.cpp)
add_dependencies(dataset_slam ${PROJECT_NAME}_gencfg)
target_link_libraries(dataset_slam 
  ${catkin_LIBRARIES} 
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}  
  ${Eigen3_LIBRARIES}  
  lsd_slam_core)

