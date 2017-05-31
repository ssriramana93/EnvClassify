/* Include the controller definition */
#include "footbot_diffusion.h"
/* Function definitions for XML parsing */
#include <argos3/core/utility/configuration/argos_configuration.h>
/* 2D vector definition */
#include <argos3/core/utility/math/vector2.h>

/****************************************/
/****************************************/

void findMinCluster(std::vector<CVector>& cVec, std::tuple<size_t,size_t>& minIdx) {
   bool begflag = true; 
   struct compIdx
   {
      bool operator()(const std::tuple<size_t,size_t>& s1, const std::tuple<size_t,size_t>& s2) const
      {
        return ((std::get<1>(s1) - std::get<0>(s1)) < (std::get<1>(s2) - std::get<0>(s2)));
      }
   };
   std::set<std::tuple<size_t,size_t>, compIdx > index_counts;
   size_t beg = 0, end = 0, n = 0;
   for(size_t i=0; i < cVec.size(); ++i) {
    if(cVec.Length() < m_fDelta) {
      if(begflag) {
      beg = i;
      end = i;
      begflag = false;
    }
    else {
    end++;
  }
  }
    else {
      if(!begflag) {
        index_counts.insert(std::make_tuple(beg,end))
      }
      begflag = true;
    }
   }

   minIdx = *index_counts.begin();
}



CFootBotDiffusion::CFootBotDiffusion() :
   m_pcWheels(NULL),
   m_pcProximity(NULL),
   m_cAlpha(10.0f),
   m_fDelta(0.5f),
   m_fWheelVelocity(2.5f),
   rotcount(0),
   lrot(true),
   temp_vel(m_fWheelVelocity),
   m_cGoStraightAngleRange(-ToRadians(m_cAlpha),
                           ToRadians(m_cAlpha)) {}

/****************************************/
/****************************************/

void CFootBotDiffusion::Init(TConfigurationNode& t_node) {
   /*
    * Get sensor/actuator handles
    *
    * The passed string (ex. "differential_steering") corresponds to the
    * XML tag of the device whose handle we want to have. For a list of
    * allowed values, type at the command prompt:
    *
    * $ argos3 -q actuators
    *
    * to have a list of all the possible actuators, or
    *
    * $ argos3 -q sensors
    *
    * to have a list of all the possible sensors.
    *
    * NOTE: ARGoS creates and initializes actuators and sensors
    * internally, on the basis of the lists provided the configuration
    * file at the <controllers><footbot_diffusion><actuators> and
    * <controllers><footbot_diffusion><sensors> sections. If you forgot to
    * list a device in the XML and then you request it here, an error
    * occurs.
    */
   m_pcWheels    = GetActuator<CCI_DifferentialSteeringActuator>("differential_steering");
   m_pcProximity = GetSensor  <CCI_FootBotProximitySensor      >("footbot_proximity"    );
   /*
    * Parse the configuration file
    *
    * The user defines this part. Here, the algorithm accepts three
    * parameters and it's nice to put them in the config file so we don't
    * have to recompile if we want to try other settings.
    */
   GetNodeAttributeOrDefault(t_node, "alpha", m_cAlpha, m_cAlpha);
   m_cGoStraightAngleRange.Set(-ToRadians(m_cAlpha), ToRadians(m_cAlpha));
   GetNodeAttributeOrDefault(t_node, "delta", m_fDelta, m_fDelta);
   GetNodeAttributeOrDefault(t_node, "velocity", m_fWheelVelocity, m_fWheelVelocity);
}

/****************************************/
/****************************************/

void CFootBotDiffusion::ControlStep() {
   /* Get readings from proximity sensor */
   const CCI_FootBotProximitySensor::TReadings& tProxReads = m_pcProximity->GetReadings();
   size_t rotthresh = 1000;
   std::vector<CVector2> obsinfront, obsvector;

   /* Sum them together */
   CVector2 cAccumulator,cMax;
   int nobsSt = 0,nobsLeft = 0,nobsRight = 0,nobsBehind = 0;
   for(size_t i = 0; i < tProxReads.size(); ++i) {
       CVector2 cTemp = CVector2(tProxReads[i].Value, tProxReads[i].Angle);
       obsvector.append(cTemp);
       if(((i>1) && (i < 6)) || ((i<24) && (i>19)))  {
       /*if (m_cGoStraightAngleRange.WithinMinBoundIncludedMaxBoundIncluded(cTemp.Angle()) && !(cTemp.Length() < m_fDelta))*/ 
        if(cTemp.Length() > m_fDelta) {
           nobsSt++;
         }  
         obsinfront.append(cTemp);
    //    if(cMax.Length() < cTemp.Length()) {
    //      cMax = cTemp;
    //    }
    //    cAccumulator += cTemp; 
       }
       
       if((i>2) && (i<12)) {
         if(cTemp.Length() > m_fDelta) {
           nobsLeft++;
         }  

       }

       if((i>14) && (i<23)) {
        if(cTemp.Length() > m_fDelta) {
           nobsLeft++;
         }  
       }

        if((i>7) && (i<18)) {
        if(cTemp.Length() > m_fDelta) {
           nobsBehind++;
         }  
       }
    
   }
   //cAccumulator /= nobs;
   cAccumulator /= tProxReads.size();
   /* If the angle of the vector is small enough and the closest obstacle
    * is far enough, continue going straight, otherwise curve a little
    */
  // CRadians cAngle = cMax.Angle();

  // std::cout<<"nobs"<<nobs<<std::endl;
   //        m_pcWheels->SetLinearVelocity(-temp_vel, -temp_vel);

   if(!nobsSt) {
     // rotcount = 0;
     // temp_vel = m_fWheelVelocity;

      /* Go straight */
      m_pcWheels->SetLinearVelocity(m_fWheelVelocity, m_fWheelVelocity);
   }
   else {
      /* Turn, depending on the sign of the angle */
 //     if(cAngle.GetValue() > 0.0f) {
        if(!nobsRight) {
       /*  if (lrot) {
           rotcount++;
         }
         if (lrot && (rotcount > rotthresh)) {
        //  temp_vel /= 2;
          rotcount = 0;
         }
         lrot = true;*/
         m_pcWheels->SetLinearVelocity(m_fWheelVelocity, -m_fWheelVelocity);

      }
      else if(!nobsLeft){
       /*  if (!lrot) {
          rotcount++;
         }
         if (!lrot && (rotcount > rotthresh)) {
        //   temp_vel /= 2;
           rotcount = 0;
         }
         lrot = false;*/
         m_pcWheels->SetLinearVelocity(-m_fWheelVelocity, m_fWheelVelocity);
      }

      else if(!nobsBehind) {

        m_pcWheels->SetLinearVelocity(-m_fWheelVelocity, -m_fWheelVelocity);
      }

      else {
         std::tuple<size_t, size_t> minIdx;
         findMinCluster(obsvector, minIdx);
         auto cMinVec = obsvector[(int)((std::get<1>(minIdx) + std::get<0>(minIdx))/2)];
         cAngle = cMinVec.Angle();
         if(cAngle.GetValue() > 0.0f) {
          m_pcWheels->SetLinearVelocity(m_fWheelVelocity/2, -m_fWheelVelocity/2);
         }
         else {
          m_pcWheels->SetLinearVelocity(-m_fWheelVelocity/2, m_fWheelVelocity/2);
         }
       }
       //  m_pcWheels->SetLinearVelocity(m_fWheelVelocity, m_fWheelVelocity);
      }
   }
}

/****************************************/
/****************************************/

/*
 * This statement notifies ARGoS of the existence of the controller.
 * It binds the class passed as first argument to the string passed as
 * second argument.
 * The string is then usable in the configuration file to refer to this
 * controller.
 * When ARGoS reads that string in the configuration file, it knows which
 * controller class to instantiate.
 * See also the configuration files for an example of how this is used.
 */
REGISTER_CONTROLLER(CFootBotDiffusion, "footbot_diffusion_controller")
