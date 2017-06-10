CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "DEBUG")
  #SET(CMAKE_BUILD_TYPE "RELEASE")
  #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()
#SET (BOOST_ROOT "/opt/boost/boost_1_58_0")
#SET (BOOST_INCLUDEDIR "/opt/boost/boost-1.58.0/include")
#SET (BOOST_LIBRARYDIR "/opt/boost/boost-1.58.0/lib")


SET (BOOST_MIN_VERSION "1.55.0")
set (Boost_NO_BOOST_CMAKE ON)
FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED)
find_package( PythonLibs 2.7 REQUIRED )
find_package( PythonInterp 2.7 REQUIRED )
#FIND_PACKAGE(Boost 1.58.0)
IF(Boost_FOUND)
  INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" "/usr/include/python2.7")
  SET(Boost_USE_STATIC_LIBS OFF)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME OFF)
  FIND_PACKAGE(Boost 1.58.0 COMPONENTS python)

  ADD_LIBRARY(experiment SHARED experiment.hpp experiment.cpp)
  TARGET_LINK_LIBRARIES(experiment 
footbot_envclassify_tf
  envclassify_loop_functions_tf 
  argos3core_simulator
  ${Boost_LIBRARIES}
  )
  set_target_properties(experiment PROPERTIES
 LIBRARY_OUTPUT_DIRECTORY ~/argos3-examples/python_code)
ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-Wall")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()
