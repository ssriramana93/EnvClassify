add_executable(envclassify main.cpp)

target_link_libraries(envclassify
  ${GALIB_LIBRARIES}
  argos3core_simulator
  footbot_envclassify
  envclassify_loop_functions)
