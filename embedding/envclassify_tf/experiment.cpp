#include <experiment.hpp>
#include <argos3/core/simulator/simulator.h>
#include <argos3/core/simulator/loop_functions.h>

#include <loop_functions/envclassify_loop_functions_tf/envclassify_loop_functions_tf.h>

double experiment::LaunchARGoS() {
   /* Convert the received genome to the actual genome type */
 //  GARealGenome& cRealGenome = dynamic_cast<GARealGenome&>(c_genome);
   /* The CSimulator class of ARGoS is a singleton. Therefore, to
    * manipulate an ARGoS experiment, it is enough to get its instance.
    * This variable is declared 'static' so it is created
    * once and then reused at each call of this function.
    * This line would work also without 'static', but written this way
    * it is faster. */
   static argos::CSimulator& cSimulator = argos::CSimulator::GetInstance();
   /* Get a reference to the loop functions */
   static EnvClassifyLoopFunctions& cLoopFunctions = dynamic_cast<EnvClassifyLoopFunctions&>(cSimulator.GetLoopFunctions());
   /*
    * Run 5 trials and take the worst performance as final value.
    * Performance in this experiment is defined as the distance from the light.
    * Thus, we keep the max distance found.
    */
   Real fDistance = std::numeric_limits<Real>::max();

   for(size_t i = 0; i < 10; ++i) {
      /* Tell the loop functions to get ready for the i-th trial */
      cLoopFunctions.SetTrial(i);
      /* Reset the experiment.
       * This internally calls also CEvolutionLoopFunctions::Reset(). */
      cSimulator.Reset();

      //@todo: Need to do something
      /* Configure the controller with the genome */
 //     cLoopFunctions.ConfigureFromGenome(cRealGenome);
   //   std::cout<<"Execute Called" <<std::endl;
      /* Run the experiment */
      cSimulator.Execute();

  //    std::cout<<"Execute Terminated" <<std::endl;
      /* Update performance */

      fDistance = Min(fDistance, cLoopFunctions.Performance());
     // std::cout<<"fDistance"<<fDistance;
   }
   /* Return the result of the evaluation */
   return fDistance;
}




double experiment::execute() {
   /*
    * Initialize ARGoS
    */
   /* The CSimulator class of ARGoS is a singleton. Therefore, to
    * manipulate an ARGoS experiment, it is enough to get its instance */
   argos::CSimulator& cSimulator = argos::CSimulator::GetInstance();
   /* Set the .argos configuration file
    * This is a relative path which assumed that you launch the executable
    * from argos3-examples (as said also in the README) */
   cSimulator.SetExperimentFileName("python_code/envclassify.argos");
   /* Load it to configure ARGoS */
   cSimulator.LoadExperiment();

   return LaunchARGoS();

}

using namespace boost::python;

BOOST_PYTHON_MODULE(libexperiment)
{
    class_<experiment>("experiment")
        .def("execute", &experiment::execute)
        .def("launchArgos", &experiment::LaunchARGoS)

    ;
}
/****************************************/
/****************************************/

/****************************************/
/****************************************/
