#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "neural_network.h"

class CPerceptron : public CNeuralNetwork {

public:

   CPerceptron();
   virtual ~CPerceptron();
  
   virtual void Init(TConfigurationNode& t_tree);
   virtual void Destroy();

   virtual void LoadNetworkParameters(const std::string& str_filename );
   virtual void LoadNetworkParameters(const UInt32 un_num_params,
                                      const Real* pf_params );
   virtual void ComputeOutputs();  

   bool isWeightInit();

   void printAllVals() {
      for(size_t  i = 0; i < m_unNumberOfOutputs;i++) {
      std::cout<<m_pfOutputs[i]<<std::endl;
     }
      for(size_t  i = 0; i < m_unNumberOfInputs;i++) {
      std::cout<<m_pfInputs[i]<<std::endl;
       }

       for(size_t  i = 0; i < m_unNumberOfWeights;i++) {
      std::cout<<m_pfWeights[i]<<std::endl;
       }
       
   }

private:

   UInt32   m_unNumberOfWeights;
   Real*    m_pfWeights;
  
};

#endif
