cmake_minimum_required(VERSION 2.8)
project(test_component)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

set (CMAKE_CXX_FLAGS "-O2")

set(CMAKE_BUILD_TYPE Release)

include_directories(
	${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
	)

link_directories(${PCL_LIBRARY_DIRS}) 
add_definitions( ${PCL_DEFINITIONS} )

add_executable(test_pcl
	pcl_io_show.cpp
	CAPE.cpp
	PlaneSeg.cpp
	CylinderSeg.cpp
	Histogram.cpp)

target_link_libraries(test_pcl 
	${OpenCV_LIBS} 
	${Eigen_LIBRARIES}
	${PCL_LIBRARIES}
	)
