add_executable(galib_phototaxis_test main.cpp)

target_link_libraries(galib_phototaxis_test
  ${GALIB_LIBRARIES}
  footbot_nn_test
  galib_phototaxis_loop_functions_test
  argos3core_simulator)
