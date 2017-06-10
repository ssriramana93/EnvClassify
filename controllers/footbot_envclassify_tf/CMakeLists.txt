add_library(footbot_envclassify_tf SHARED
  nn/rnn.h
  nn/rnn.cpp
  
  footbot_rnn_controller.h
  footbot_rnn_controller.cpp
)
target_link_libraries(footbot_envclassify_tf
  argos3core_simulator
  argos3plugin_simulator_footbot
  argos3plugin_simulator_genericrobot)
