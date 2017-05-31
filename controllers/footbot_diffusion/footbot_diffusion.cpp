/* Include the controller definition */
#include "footbot_diffusion.h"
/* Function definitions for XML parsing */
#include <argos3/core/utility/configuration/argos_configuration.h>
/* 2D vector definition */
#include <argos3/core/utility/math/vector2.h>
#include <argos3/core/utility/logging/argos_log.h>

#include <argos3/core/utility/math/rng.h>
#include <argos3/core/utility/math/range.h>


#include <vector>
#include <tuple>
#include <set>

using namespace argos;
/****************************************/
/****************************************/

void findMinCluster(std::vector<CVector2>& cVec, std::tuple<size_t,size_t>& minIdx, Real m_fDelta) {
   bool begflag = true; 
   std::cout<<"AA"<<std::endl;
   struct compIdx
   {
      bool operator()(const std::tuple<size_t,size_t>& s1, const std::tuple<size_t,size_t>& s2) const
      {
        return ((std::get<1>(s1) - std::get<0>(s1)) > (std::get<1>(s2) - std::get<0>(s2)));
      }
   };
   std::cout<<"BB"<<std::endl;

   std::set<std::tuple<size_t,size_t>, compIdx > index_counts;
   size_t beg = 0, end = 0, n = 0;
   for(size_t i=0; i < cVec.size(); ++i) {
    std::cout<<"\ti=\t"<<i<<"\tLen=\t"<<cVec[i].Length()<<std::endl;
    if(cVec[i].Length() < m_fDelta) {
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
        index_counts.insert(std::make_tuple(beg,end));
      }
      begflag = true;
    }
   }
      std::cout<<"CC"<<std::endl;

   minIdx = *index_counts.begin();
}



CFootBotDiffusion::CFootBotDiffusion() :
   m_pcWheels(NULL),
   m_pcProximity(NULL),
   m_cAlpha(10.0f),
   m_fDelta(0.5f),
   m_fWheelVelocity(2.5f),
   m_cGoStraightAngleRange(-ToRadians(m_cAlpha),
                           ToRadians(m_cAlpha)),
   crng(1111) {}

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

   podd = std::make_tuple(0,0,0,0);
   peven = std::make_tuple(0,0,0,0);
}

/****************************************/
/****************************************/
void CFootBotDiffusion::moveleft(Real scale = 1.0) {
      if (!integration_disable) {
     wheel_movements.push_back(std::make_pair(-m_fWheelVelocity*scale, m_fWheelVelocity*scale));
   }
     m_pcWheels->SetLinearVelocity(-m_fWheelVelocity*scale, m_fWheelVelocity*scale);
}

void CFootBotDiffusion::moveright(Real scale = 1.0) {
  if (!integration_disable) {
     wheel_movements.push_back(std::make_pair(m_fWheelVelocity*scale, -m_fWheelVelocity*scale));
   }
     m_pcWheels->SetLinearVelocity(m_fWheelVelocity*scale, -m_fWheelVelocity*scale);
}

void CFootBotDiffusion::movestraight(Real scale = 1.0) {
  if (!integration_disable) {
     wheel_movements.push_back(std::make_pair(m_fWheelVelocity*scale, m_fWheelVelocity*scale));
   }
     m_pcWheels->SetLinearVelocity(m_fWheelVelocity*scale, m_fWheelVelocity*scale);
}

void CFootBotDiffusion::movebehind(Real scale = 1.0) {
  if (!integration_disable) {
     wheel_movements.push_back(std::make_pair(-m_fWheelVelocity*scale, -m_fWheelVelocity*scale));
   }
     m_pcWheels->SetLinearVelocity(-m_fWheelVelocity*scale, -m_fWheelVelocity*scale);
}


void CFootBotDiffusion::ControlStep() {
   /* Get readings from proximity sensor */
   const CCI_FootBotProximitySensor::TReadings& tProxReads = m_pcProximity->GetReadings();
   std::vector<CVector2> obsinfront, obsvector;

   int nobsSt = 0,nobsLeft = 0,nobsRight = 0,nobsBehind = 0;
   for(size_t i = 0; i < tProxReads.size(); ++i) {
       CVector2 cTemp = CVector2(tProxReads[i].Value, tProxReads[i].Angle);
       obsvector.push_back(cTemp);
       //Straight Quadrant
       if(((i>1) && (i < 6)) || ((i<24) && (i>19)))  {
        if(cTemp.Length() > m_fDelta) {
           nobsSt++;
         }  
         obsinfront.push_back(cTemp);
       }
       
       //Left Quadrant
       if((i>2) && (i<12)) {
         if(cTemp.Length() > m_fDelta) {
           nobsLeft++;
         }  

       }


       if((i>14) && (i<23)) {
        if(cTemp.Length() > m_fDelta) {
           nobsRight++;
         }  
       }

        if((i>10) && (i<15)) {
        if(cTemp.Length() > m_fDelta) {
           nobsBehind++;
         }  
       }
    
   }


   

   if (backtrack_count) {
    integration_disable = true;
    wheel_movements.clear();
   //  if (backtrack_count <= 50) {
    CRange<Real> range(0.0f,1.0f);
    auto rnum = crng.Uniform(range);
    RLOG<<"rnum"<<rnum<<std::endl;
    if ((rnum >= 0) && (rnum < 0.3)) {
          RLOG<<"nobsLeft"<<nobsLeft<<std::endl;

      if(!nobsLeft) {
        moveleft(2.0);
       } 
    }
    else if((rnum >= 0.3) && (rnum < 0.6)) {
       RLOG<<"nobsRight"<<nobsLeft<<std::endl;

       if(!nobsRight) {

        moveright(2.0);
      }
    }
     
    else {
      if(!nobsBehind) {
     movebehind();
    }
    }
    backtrack_count--;

    return;
   }
   integration_disable = false;

   if(!wheel_movements.empty()) {
   if(wheel_movements.size() >= integration_window) {
    wheel_movements.erase(wheel_movements.begin());
   
   std::pair<Real, Real> integral(0.0f,0.0f);
   for (size_t i = 0; i < integration_window; i++) {
    auto movement = wheel_movements[i];
    integral.first += movement.first;
    integral.second += movement.second;

   }
   RLOG<<"Int0 = "<<integral.first<<"\tInt1 = "<<integral.second<<std::endl;
   if((std::abs(integral.first) < integration_window) && (std::abs(integral.second) < integration_window)) {
    stuckcount++;
   }
 }
 }

   
  
 /*  size_t oddiff = 0, evendiff = 0;
   if(!alt) {
    oddiff = (std::get<0>(podd) == nobsSt) && (std::get<1>(podd) == nobsRight) && (std::get<2>(podd) == nobsLeft) && (std::get<3>(podd) == nobsBehind);
    podd = std::make_tuple(nobsSt,nobsRight,nobsLeft,nobsBehind);

   }
   else {
    evendiff = (std::get<0>(peven) == nobsSt) && (std::get<1>(peven) == nobsRight) && (std::get<2>(peven) == nobsLeft) && (std::get<3>(peven) == nobsBehind);
    peven = std::make_tuple(nobsSt,nobsRight,nobsLeft,nobsBehind);
   }

   alt = !alt;
   RLOG<<"nobsSt="<<nobsSt<<"nobsLeft="<<nobsLeft<<"nobsRight"<<nobsRight<<"nobsBehind"<<nobsBehind<<std::endl;

   RLOG<<"oddiff="<<oddiff<<"evendiff="<<evendiff<<"podd!=peven"<<(podd != even)<<std::endl;

   if((oddiff) && (evendiff) && (podd != peven)) {
    stuckcount++;
   }
   else{
    stuckcount = 0;
   }
   RLOG<<"SC"<<stuckcount<<std::endl;
   */
   if(stuckcount) {
    backtrack_count = 100;
    stuckcount = 0;
    return;
   }


   //nobs.append(std::make_tuple(nobsSt,nobsRight,nobsLeft,nobsBehind));



   if(!nobsSt) {
    
      movestraight();
  //    prevR = prevL = false;
  //    stuckcount = 0;

   }
   else {

 
      if(!nobsRight) {
       
         moveright();
    /*     if(!prevR) {
          prevR = true;
         }
         if(prevL) {
          stuckcount++;
         }
         if(stuckcount > 10) {
          backtrack_count = 10;
          stuckcount = 0;
         }*/

      }

       

      else if(!nobsLeft){
      
         moveleft();
       /*  if(!prevL) {
          prevL = true;

         }
         if(prevR) {
          stuckcount++;
         }
         if(stuckcount > 10) {
          backtrack_count = 10;
          stuckcount = 0;
         }*/
      }


      else if(!nobsBehind) {

       movebehind();
      //  prevR = prevL = false;
      //  stuckcount = 0;

      }
     

      else {
      //   prevR = prevL = false;
      //   stuckcount = 0;

         std::tuple<size_t, size_t> minIdx;
         findMinCluster(obsvector, minIdx, m_fDelta);

         size_t minIndex = (size_t)((std::get<1>(minIdx) + std::get<0>(minIdx))/2);
         RLOG<<minIndex<<std::endl;

         if((minIndex>7) && (minIndex<18)) {
          movebehind();
           backtrack_count = 100;

           return;
         }           
         auto cMinVec = obsvector[minIndex];
         auto cAngle = cMinVec.Angle();
         if(cAngle.GetValue() > 0.0f) {
          moveleft();
         }
         else {
          moveright();
         }
        //m_pcWheels->SetLinearVelocity(2*m_fWheelVelocity, 2*-m_fWheelVelocity);

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
