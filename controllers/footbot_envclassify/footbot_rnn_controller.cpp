
#include "footbot_rnn_controller.h"
#include <argos3/core/utility/logging/argos_log.h>


/****************************************/
/****************************************/

static CRange<Real> NN_OUTPUT_RANGE(0.0f, 1.0f);
static CRange<Real> WHEEL_ACTUATION_RANGE(-5.0f, 5.0f);
static CRange<Real> COMM_DATA_RANGE(0.0f, 255.0f);

/****************************************/
/****************************************/

CFootBotRNNController::CFootBotRNNController():
maxMessagesAllowed(3) {
  std::cout<<"Controller Created"<<std::endl;
}

/****************************************/
/****************************************/

CFootBotRNNController::~CFootBotRNNController() {
  std::cout<<"Destructor for RNN"<<std::endl;

}

/****************************************/
/****************************************/

void CFootBotRNNController::Init(TConfigurationNode& t_node) {
   /*
    * Get sensor/actuator handles
    */
   try {
      std::cout<<"Controller Initialized"<<std::endl;
      m_pcWheels    = GetActuator<CCI_DifferentialSteeringActuator>("differential_steering");
      m_pcProximity = GetSensor  <CCI_FootBotProximitySensor      >("footbot_proximity"    );
      m_pcRABA      = GetActuator<CCI_RangeAndBearingActuator     >("range_and_bearing"    );
      m_pcRABS      = GetSensor  <CCI_RangeAndBearingSensor       >("range_and_bearing"    );
      m_pcGround    = GetSensor  <CCI_FootBotMotorGroundSensor    >("footbot_motor_ground" );
      commData = new uint8_t[maxDataSize];
      std::cout<<"All done"<<std::endl;


   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing sensors/actuators", ex);
   }

   /* Initialize the perceptron */
   try {
      m_cRNN.Init(t_node);
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing the perceptron network", ex);
   }
   std::cout<<"All done"<<std::endl;

}

/****************************************/
/****************************************/

void CFootBotRNNController::ControlStep() {
   //RLOG<<"Controller ControlStep"<<std::endl;
//   std::cout<<"Controller Control Step"<<std::endl;

   /* Get sensory data */
   const CCI_FootBotProximitySensor::TReadings& tProx = m_pcProximity->GetReadings();
   const CCI_FootBotMotorGroundSensor::TReadings& tGroundReads = m_pcGround->GetReadings();
   const CCI_RangeAndBearingSensor  ::TReadings& tRABS = m_pcRABS->GetReadings();
   

   std::vector<Real> cAccumulateInput;
   /* Fill NN inputs from sensory data */
   for(size_t i = 0; i < tProx.size(); ++i) {
      cAccumulateInput.push_back(tProx[i].Value);
   }
   //std::cout<<"A"<<std::endl;

   for(size_t i = 0; i < tGroundReads.size(); ++i) {
      cAccumulateInput.push_back(tGroundReads[i].Value);
   }
     // std::cout<<"B"<<std::endl;

   try {
      feedComToNN(tRABS,cAccumulateInput);
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error feeding comm to input accumulator", ex);
   }

   try {
     for(size_t i = 0; i < cAccumulateInput.size(); i++) {
      m_cRNN.SetInput(i, cAccumulateInput[i]);
   }
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error feeding input to perceptron", ex);
   }
    //std::cout<<"C"<<std::endl;

   /* Compute NN outputs */
   m_cRNN.ComputeOutputs();
   //    std::cout<<"D"<<std::endl;

   /*
    * Apply NN outputs to actuation
    * The NN outputs are in the range [0,1]
    * To allow for backtracking, we remap this range
    * into [-5:5] linearly.
    */
   NN_OUTPUT_RANGE.MapValueIntoRange(
      m_fLeftSpeed,               // value to write
      m_cRNN.GetOutput(0), // value to read
      WHEEL_ACTUATION_RANGE       // target range (here [-5:5])
      );
   NN_OUTPUT_RANGE.MapValueIntoRange(
      m_fRightSpeed,              // value to write
      m_cRNN.GetOutput(1), // value to read
      WHEEL_ACTUATION_RANGE       // target range (here [-5:5])
      );
   m_pcWheels->SetLinearVelocity(
      m_fLeftSpeed,
      m_fRightSpeed);
   //std::cout<<"LS"<<m_fLeftSpeed<<"RS"<<m_fRightSpeed<<std::endl;


   for (size_t i = 0; i < maxDataSize; i++) {
   Real data;
   NN_OUTPUT_RANGE.MapValueIntoRange(
      data,              // value to write
      m_cRNN.GetOutput(i + 2), // value to read
      COMM_DATA_RANGE       // target range (here [-5:5])
      );
   commData[i] = static_cast<uint8_t>(data);
  // std::cout<<"commData ="<<commData[i]<<"Data="<<data<<std::endl;
   }
  //   std::cout<<"F"<<std::endl;

   for(size_t  i = 0; i < maxDataSize; i++)
        m_pcRABA->SetData(i,commData[i]);

   envProbVec = {m_cRNN.GetOutput(maxDataSize + 2),m_cRNN.GetOutput(maxDataSize + 3),m_cRNN.GetOutput(maxDataSize + 4)};
   envProbVecs_.push_back(envProbVec);
  // std::cout<<"maxDataSize"<<maxDataSize + 3<<std::endl; 
   

}

/****************************************/
/****************************************/

void CFootBotRNNController::Reset() {
//  std::cout<<"Reset Called in RNN"<<std::endl;
   m_cRNN.Reset();
   m_pcRABA->ClearData();
   envProbVecs_.clear();
}

/****************************************/
/****************************************/

void CFootBotRNNController::Destroy() {
 //   std::cout<<"Destroy Called in RNN"<<std::endl;

   m_cRNN.Destroy();
}

void CFootBotRNNController::feedComToNN(const CCI_RangeAndBearingSensor::TReadings& tRABS, std::vector<Real>& cAccumulateInput) {
  //@todo: Feed in constant number of readings and select dummy values when enough data is not present
//  std::cout<<"FeedComm"<<tRABS.size()<<std::endl;
  size_t datasize = maxDataSize;
  for(size_t i = 0; i < maxMessagesAllowed; ++i) {
  //  std::cout<<"AA"<<i<<std::endl;
      if (i >= tRABS.size()) {
        cAccumulateInput.push_back(0.0f);
        cAccumulateInput.push_back(0.0f);
        cAccumulateInput.push_back(0.0f);
        for(size_t i = 0;i < datasize; i++) { cAccumulateInput.push_back(0.0f);}
        continue;
      }
      cAccumulateInput.push_back(tRABS[i].Range);
      cAccumulateInput.push_back(tRABS[i].HorizontalBearing.GetValue());
      cAccumulateInput.push_back(tRABS[i].VerticalBearing.GetValue());
      auto data_array = tRABS[i].Data.ToCArray();
         for(size_t  i = 0; i < maxDataSize; i++)
             { cAccumulateInput.push_back(static_cast<Real>(data_array[i])); }
   }
 // std::cout<<"FeedCommDone"<<std::endl;


}

/****************************************/
/****************************************/

REGISTER_CONTROLLER(CFootBotRNNController, "footbot_rnn_controller")
