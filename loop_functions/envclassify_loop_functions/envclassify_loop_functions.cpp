
#include <envclassify_loop_functions.h>
#include <argos3/core/simulator/simulator.h>
#include <argos3/core/utility/configuration/argos_configuration.h>
#include <argos3/plugins/robots/foot-bot/simulator/footbot_entity.h>
#include <controllers/footbot_envclassify/footbot_rnn_controller.h>
#include <argos3/core/utility/logging/argos_log.h>


#include <sstream> 

/****************************************/
/****************************************/

EnvClassifyLoopFunctions::EnvClassifyLoopFunctions() :
   score(0.0f),
   nEnvs(20),
   xLength(100),
   yLength(100),
   m_pcFloor(NULL),
   m_pcRNG(NULL),
   pixelPerMeter(20),
   nFootBots(5),
   m_unCurrentTrial(0),
   m_pfControllerParams(new Real[GENOME_SIZE]) {
    std::cout<<"EnvClassifyLoopFunctions Object Created..."<<std::endl;
}

EnvClassifyLoopFunctions::~EnvClassifyLoopFunctions() {
 // std::cout<<"Destructor Called"<<std::endl;
  delete[] m_pfControllerParams;
 /* for (auto const &iter1: envList) {
     for (auto const &iter2: iter1) {
       delete[] iter2;
      }
     delete[] iter1; 
  }*/
  for (size_t i = 0; i < m_pcFootBots.size(); i++) {
    delete m_pcFootBots[i];
    delete m_pcControllers[i];
  }
  m_pcFootBots.clear();
  m_pcControllers.clear();
}

/****************************************/
/****************************************/

void EnvClassifyLoopFunctions::Init(TConfigurationNode& t_node) {
   try {
  //    std::cout<<"EnvClassifyLoopFunctions Init Called..."<<std::endl;

      /* Get a pointer to the floor entity */
      m_pcFloor = &GetSpace().GetFloorEntity();
      m_pcArenaLimits = GetSpace().GetArenaLimits();
      ArenaSize = GetSpace().GetArenaSize();

 //     std::cout<<"Arena Size X= "<<ArenaSize[0]<<" Y= "<<ArenaSize[1]<<std::endl;

      GetNodeAttributeOrDefault(t_node, "nFootBots", nFootBots, nFootBots);
      GetNodeAttributeOrDefault(t_node, "pixels_per_meter", pixelPerMeter, pixelPerMeter);


      xLength = static_cast<size_t>(pixelPerMeter*ArenaSize[0]);
      yLength = static_cast<size_t>(pixelPerMeter*ArenaSize[1]);

   //   std::cout<<"xLength = "<<xLength<<" yLength= "<<yLength<<std::endl;

   
      m_pcRNG = CRandom::CreateRNG("argos");
      for (size_t i = 0; i < nFootBots; i++) {
     //   std::cout<<"Creating Footbot "<<i<<std::endl;
        std::ostringstream os;
        os<<"fb"<<i<<std::endl;  
        /*auto m_pcFootBot = std::make_shared<CFootBotEntity>(
        os.str(),    // entity id
        "fnn"    // controller id as set in the XML
        );*/

        auto m_pcFootBot = new CFootBotEntity(
        os.str(),    // entity id
        "fnn"    // controller id as set in the XML
        );
        AddEntity(*m_pcFootBot);
        m_pcFootBots.push_back(m_pcFootBot);
       // m_pcControllers.push_back(std::make_shared<CFootBotRNNController>(dynamic_cast<CFootBotRNNController&>(m_pcFootBot->GetControllableEntity().GetController())));
        m_pcControllers.push_back(&dynamic_cast<CFootBotRNNController&>(m_pcFootBot->GetControllableEntity().GetController()));
      }
      
   //   std::cout<<"Footbots creation successful.."<<std::endl;

      for (size_t i = 0; i < nEnvs; i++) {
      //  std::cout<<"Generating Env "<<i<<std::endl;
          size_t tol = 10;

         PixToColor pixtocolor(xLength + tol, std::vector<CColor>(yLength + tol));

       //  pixtocolor = std::make_unique<std::unique_ptr<CColor> >(new std::unique_ptr<CColor>[xLength]);
      //   pixtocolor = new CColor*[xLength];
         for(int i = 0; i < (xLength + tol); ++i) {
       //     pixtocolor[i] = std::make_unique<CColor[]>(new CColor[yLength]);
         //  pixtocolor[i] = new CColor[yLength];
            for(size_t j = 0; j< (yLength + tol); j++) 
               pixtocolor[i][j] = CColor::WHITE;
       }
       // Even = Uniform, Odd = Gaussian
         if((i % 2)==1) {
            GenGaussianEnv(0.1, 1000, pixtocolor);
         }
         else {
            GenUniformEnv(3000, pixtocolor);
         }
         envList.push_back(pixtocolor);
      }
      
//       std::cout<<"Env Generation successful.."<<std::endl;

   
       GetNodeAttributeOrDefault(t_node, "trial", m_unCurrentTrial, m_unCurrentTrial);
       Reset();
    

   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error parsing loop functions!",ex);
   }
}



/****************************************/
/****************************************/

void EnvClassifyLoopFunctions::Reset() {
//  std::cout<<"EnvClassifyLoopFunctions Reset Called..."<<std::endl;
  CVector3 lLimit = m_pcArenaLimits.GetMin();
  //std::cout<<"Seed"<<seed<<std::endl;
 // m_pcRNG->SetSeed(seed);


  for (auto const footbot: m_pcFootBots) {
  
  //bool created = false;  
  while(1) {  
  //std::cout<<"ArenaLimits ("<<lLimit[0]<<","<<lLimit[1]<<")"<<std::endl;
  CVector3 Position;
  Position.SetX(lLimit[0] + m_pcRNG->Uniform(CRange<Real>(0.5f,ArenaSize[0] - 0.5f)));
  Position.SetY(lLimit[1] + m_pcRNG->Uniform(CRange<Real>(0.5f,ArenaSize[1] - 0.5f)));
//  std::cout<<"R: "<<lLimit[0] + m_pcRNG->Uniform(CRange<Real>(0.5f,ArenaSize[0] - 0.5f))<<std::endl;
  CRadians cOrient = m_pcRNG->Uniform(CRadians::UNSIGNED_RANGE);
  CQuaternion Orientation;
  Orientation.FromEulerAngles(
         cOrient,        // rotation around Z
         CRadians::ZERO, // rotation around Y
         CRadians::ZERO  // rotation around X
         );
  if(!MoveEntity(
         footbot->GetEmbodiedEntity(),             // move the body of the robot
         Position,    // to this position
         Orientation, // with this orientation
         false                                         // this is not a check, leave the robot there
         )) {
     /* LOGERR << "Can't move robot in <"
             << Position
             << ">, <"
             << Orientation
             << ">"
             << std::endl;*/
     // continue;       

   }
   else {
    break;
   }
  }
}
  // std::cout<<"Moved All Footbots..."<<std::endl;
   //std::cout<<"CurrTrial "<<m_unCurrentTrial<<"EnvList "<<envList.size()<<std::endl;
   if (m_unCurrentTrial >= envList.size()) {
       THROW_ARGOSEXCEPTION("Trial number greater than created!");
    }
   auto trial = m_pcRNG->Uniform(CRange<UInt32>(0,nEnvs)); 
   m_unCurrentTrial = trial;
   currEnv = envList[m_unCurrentTrial];
   //currEnv = envList[trial];
   currEnvType = m_unCurrentTrial%2;
   //currEnvType = trial%2;
 //  std::cout<<"Reset Done"<<m_unCurrentTrial<<std::endl;


}

/****************************************/
/****************************************/

void EnvClassifyLoopFunctions::Destroy() {
  std::cout<<"Destroy Called"<<std::endl;
   /* Close the file */
}

/****************************************/
/****************************************/

CColor EnvClassifyLoopFunctions::GetFloorColor(const CVector2& c_position_on_plane) {
//   std::cout<<"EnvClassifyLoopFunctions GetFloorColor Called..."<<std::endl;

   /*if(c_position_on_plane.GetX() < -1.0f) {
      return CColor::GRAY50;
   }*/
   /*for(auto const &iter: currEnv) {
      if((c_position_on_plane - iter.first).SquareLength() < 0.001) {
         return iter.second;
      }
   }*/
   CVector3 lLimit = m_pcArenaLimits.GetMin(), rLimit = m_pcArenaLimits.GetMax();
   auto xIndex = static_cast<size_t>(pixelPerMeter*(c_position_on_plane.GetX() - lLimit[0])), yIndex = static_cast<size_t>(pixelPerMeter*(c_position_on_plane.GetY() - lLimit[1]));
   if(xIndex > xLength) {
    xIndex = xLength;
   } 

   if(yIndex > yLength) {
    xIndex = yLength;
  }

   CColor temp = currEnv[xIndex][yIndex];
   return temp;
}

/****************************************/
/****************************************/

void EnvClassifyLoopFunctions::GenGaussianEnv(Real f_std_dev, size_t numSpots, PixToColor& pixtocolor) {
//   std::cout<<"Generating Gaussian Env..."<<std::endl;
   Real mean_x = m_pcRNG->Uniform(CRange<Real>(0.5f,ArenaSize[0] - 0.5f));
   Real mean_y = m_pcRNG->Uniform(CRange<Real>(0.5f,ArenaSize[1] - 0.5f));


   for (size_t i = 0; i < numSpots; i++) {
       auto xIndex = static_cast<size_t>(pixelPerMeter*m_pcRNG->Gaussian(f_std_dev, mean_x)), yIndex = static_cast<size_t>(pixelPerMeter*m_pcRNG->Gaussian(f_std_dev, mean_x));

      if(xIndex > xLength) {
    xIndex = xLength;
   } 

   if(yIndex > yLength) {
    xIndex = yLength;
  }
   
      pixtocolor[xIndex][yIndex] = CColor::BLACK;

   }
}

void EnvClassifyLoopFunctions::GenUniformEnv(size_t numSpots, PixToColor& pixtocolor) {
//   std::cout<<"Generating Uniform Env..."<<std::endl;

   for (size_t i = 0; i < numSpots; i++) {
    auto xIndex = static_cast<size_t>(pixelPerMeter*m_pcRNG->Uniform(CRange<Real>(0.0f,ArenaSize[0]))), yIndex = static_cast<size_t>(pixelPerMeter*m_pcRNG->Uniform(CRange<Real>(0.0f,ArenaSize[1])));

      if(xIndex > xLength) {
    xIndex = xLength;
   } 

   if(yIndex > yLength) {
    xIndex = yLength;
  }
     
      pixtocolor[xIndex][yIndex] = CColor::BLACK;
   }
}

void EnvClassifyLoopFunctions::ConfigureFromGenome(const GARealGenome& c_genome) {
   /* Copy the genes into the NN parameter buffer */
 // std::cout<<"ConfigureFromGenome"<<std::endl;
   for(size_t i = 0; i < GENOME_SIZE; ++i) {
      m_pfControllerParams[i] = c_genome[i];
   }

	for (auto &controller: m_pcControllers) {

  
   /* Set the NN parameters */
   controller->GetRNN().SetOnlineParameters(GENOME_SIZE, m_pfControllerParams);
}

}

void EnvClassifyLoopFunctions::PreStep() {
//    std::cout<<"PreStep"<<std::endl;

}
void EnvClassifyLoopFunctions::PostStep() {
 //   std::cout<<"PostStep"<<std::endl;

}


Real EnvClassifyLoopFunctions::Performance() {
  //std::vector<EnvProbType> envProbVec;
//  std::cout<<"Performance called"<<std::endl;

  Real score = 0.0f;
  CFootBotRNNController::EnvProbsVecType envProbVecs;
  for (auto const &iter: m_pcControllers) {
 //   envProbVec.push_back(iter->GetEnvProbs());

  	iter->GetEnvProbs(envProbVecs);
    for (auto probVec: envProbVecs) {
    for(size_t  i = 0; i < 3; i++) {
    if(i == currEnvType)  {
      score += probVec[i];
    }
    else {
  //    score -= probVec[i];
      }
      }
    }
  } 
  return score;
}
/****************************************/
/****************************************/

REGISTER_LOOP_FUNCTIONS(EnvClassifyLoopFunctions, "envclassify_loop_functions")
