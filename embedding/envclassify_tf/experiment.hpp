

/* ARGoS-related headers */
#include <argos3/core/simulator/simulator.h>
#include <argos3/core/simulator/loop_functions.h>

#include <loop_functions/envclassify_loop_functions_tf/envclassify_loop_functions_tf.h>
#include <boost/python.hpp>
#include <boost/python/detail/wrap_python.hpp>
/****************************************/
/****************************************/
class experiment {
public:
	//argos::CSimulator& cSimulator;
 experiment();
 ~experiment() {
  
 	std::cout<<"experiment_d"<<std::endl;
 } 
 double LaunchARGoS();
 void execute();
 void destroy() {
   argos::CSimulator& cSimulator = argos::CSimulator::GetInstance();
   cSimulator.Destroy();
 }
};

