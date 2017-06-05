#ifndef ENVCLASSIFY_LOOP_FUNCTIONS_H
#define ENVCLASSIFY_LOOP_FUNCTIONS_H

#include <argos3/core/simulator/loop_functions.h>
#include <argos3/core/simulator/entity/floor_entity.h>
#include <argos3/plugins/robots/foot-bot/simulator/footbot_entity.h>
#include <controllers/footbot_envclassify/footbot_rnn_controller.h>
#include <argos3/core/utility/math/range.h>
#include <argos3/core/utility/math/rng.h>

#include <ga/ga.h>
#include <ga/GARealGenome.h>
#include <ga/GARealGenome.C> 

using namespace argos;
 //N = I*C1 + H*(C1**2) + V*(C2**2) + C1*C2 + C2*O
static const size_t GENOME_SIZE = 78200;

class EnvClassifyLoopFunctions : public CLoopFunctions {

public:

   EnvClassifyLoopFunctions();
   virtual ~EnvClassifyLoopFunctions();

   virtual void Init(TConfigurationNode& t_tree);
   virtual void Reset();
   virtual void Destroy();
   inline void SetTrial(size_t un_trial) {
      m_unCurrentTrial = un_trial;
     // seed = m_pcRNG->Uniform(CRange<UInt32>(0,1000));
      /*seed = static_cast<UInt32>(rand()%1000);
      std::cout<<"ST: "<<seed<<std::endl;*/
   }
   virtual CColor GetFloorColor(const CVector2& c_position_on_plane);
   virtual void PreStep();
   virtual void PostStep();

   size_t xLength, yLength; 
   size_t pixelPerMeter;

   typedef std::vector<std::vector<CColor> > PixToColor;
   //typedef CColor** PixToColor;
   //typedef std::unique_ptr<std::unique_ptr<CColor> > PixToColor;
   
   //typedef std::array<std::array<CColor, xLength>, yLength> PixToColor;
   void ConfigureFromGenome(const GARealGenome& c_genome);

   Real Performance();
private:

   Real m_fFoodSquareRadius;
   CRange<Real> m_EnvClassifyArenaSideX, m_EnvClassifyArenaSideY;
   std::vector<CVector2> m_cFoodPos;
   CFloorEntity* m_pcFloor;
   CRandom::CRNG* m_pcRNG;
   CRange< CVector3 > m_pcArenaLimits;
   CVector3 ArenaSize;

   size_t nFootBots;
  // std::vector<std::shared_ptr<CFootBotEntity> > m_pcFootBots;
   std::vector<CFootBotEntity* > m_pcFootBots;

  // std::vector<std::shared_ptr<CFootBotRNNController> > m_pcControllers;
   std::vector<CFootBotRNNController* > m_pcControllers;

   Real* m_pfControllerParams;



   std::vector<PixToColor> envList;
   PixToColor currEnv;
   size_t nEnvs;
   size_t currEnvType;

   size_t m_unCurrentTrial;

   std::string m_strOutput;
   std::ofstream m_cOutput;

   Real score;
   UInt32 seed;

   void GenGaussianEnv(Real f_std_dev, size_t numSpots, PixToColor& pixtocolor);
   void GenUniformEnv(size_t numSpots, PixToColor& pixtocolor);
   void GenScaleFreeEnv();
   


  
};

#endif
