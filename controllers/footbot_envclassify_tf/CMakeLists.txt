
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
  add_library(footbot_envclassify_tf SHARED
  	nn/rnn.h
  	nn/rnn.cpp
  
  	footbot_rnn_controller.h
  	footbot_rnn_controller.cpp
  	LSTMWrapper.cpp
  	LSTMWrapper.h

)
target_link_libraries(footbot_envclassify_tf
  argos3core_simulator
  argos3plugin_simulator_footbot
  argos3plugin_simulator_genericrobot
	${Boost_LIBRARIES})
ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()
