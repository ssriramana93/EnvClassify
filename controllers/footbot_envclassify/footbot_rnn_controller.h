#ifndef FOOTBOT_RNN_CONTROLLER
#define FOOTBOT_RNN_CONTROLLER

/*
 * Include some necessary headers.
 */
/* Definition of the CCI_Controller class. */
#include <argos3/core/control_interface/ci_controller.h>
/* Definition of the differential steering actuator */
#include <argos3/plugins/robots/generic/control_interface/ci_differential_steering_actuator.h>
/* Definition of the foot-bot proximity sensor */
#include <argos3/plugins/robots/foot-bot/control_interface/ci_footbot_proximity_sensor.h>
/* Definition of the range and bearing actuator */
#include <argos3/plugins/robots/generic/control_interface/ci_range_and_bearing_actuator.h>
/* Definition of the range and bearing sensor */
#include <argos3/plugins/robots/generic/control_interface/ci_range_and_bearing_sensor.h>
/* Definition of the foot-bot motor ground sensor */
#include <argos3/plugins/robots/foot-bot/control_interface/ci_footbot_motor_ground_sensor.h>
/* Definitions for random number generation */
#include <argos3/core/utility/math/rng.h>
/* Definition of the perceptron */
#include "nn/rnn.h"

/*
 * All the ARGoS stuff in the 'argos' namespace.
 * With this statement, you save typing argos:: every time.
 */
using namespace argos;


/*
 * A controller is simply an implementation of the CCI_Controller class.
 * In this case, we also inherit from the CPerceptron class. We use
 * virtual inheritance so that matching methods in the CCI_Controller
 * and CPerceptron don't get messed up.
 */
class CFootBotRNNController : public CCI_Controller {

public:

   CFootBotRNNController();
   virtual ~CFootBotRNNController();

   void Init(TConfigurationNode& t_node);
   void ControlStep();
   void Reset();
   void Destroy();

   inline CRNN& GetRNN() {
      return m_cRNN;
   }

//   typedef std::tuple<Real, Real, Real> EnvProbsType;
   typedef std::array<Real,3> EnvProbsType;
   typedef std::vector<EnvProbsType> EnvProbsVecType;


   void GetEnvProbs(EnvProbsVecType& envProbVecs) {
      envProbVecs = envProbVecs_;
   }

private:
   /* Pointer to the differential steering actuator */
   CCI_DifferentialSteeringActuator* m_pcWheels;
   /* Pointer to the foot-bot proximity sensor */
   CCI_FootBotProximitySensor* m_pcProximity;
   /* Pointer to the foot-bot light sensor */
   CCI_RangeAndBearingActuator*  m_pcRABA;
   /* Pointer to the range and bearing sensor */
   CCI_RangeAndBearingSensor* m_pcRABS;
    /* Pointer to the foot-bot motor ground sensor */
   CCI_FootBotMotorGroundSensor* m_pcGround;
   /* The perceptron neural network */
   CRNN m_cRNN;
   /* Wheel speeds */
   Real m_fLeftSpeed, m_fRightSpeed;

   const size_t maxDataSize = 10;
   uint8_t* commData;

   size_t maxMessagesAllowed;
   void feedComToNN(const CCI_RangeAndBearingSensor::TReadings& tRABS, std::vector<Real>& cAccumulateInput);

   EnvProbsType envProbVec;
   EnvProbsVecType envProbVecs_;

};

#endif
