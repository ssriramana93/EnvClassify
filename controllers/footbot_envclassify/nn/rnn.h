#ifndef RNN_H
#define RNN_H

#include <argos3/core/control_interface/ci_controller.h>
#include <cmath>

using namespace argos;
class CRNN  {

public:

   CRNN();
   ~CRNN();
  
   void Init(TConfigurationNode& t_tree);
   void Destroy();
   void ComputeOutputs();  

   bool isWeightInit() {return (m_pfWeights.size() == 0);}

   void LoadInputLayer(const size_t index,
                                      const Real value) {
    m_pfInputs[index] = value;
   }
  
   
   void InitNumHiddenNeurons();
   size_t GetNumWeights();
   void InitHiddenStates();
   Real GetOutput(size_t index) {
    return m_pfOutputs[index];
   }
   void SetInput(size_t index, Real value) {
    //std::cout<<"SetIpCalled: = "<<index<<"Size ="<<m_pfInputs.size()<<std::endl;
    m_pfInputs.at(index) = value;
    //m_pfInputs.push_back(value);
    //std::cout<<"SetIpCalled"<<m_pfInputs.size()<<std::endl;

   }
 
   void SetOnlineParameters(const size_t un_num_params,
                                         const Real* pf_params )  {
    if (un_num_params != m_unNumberOfWeights) {
      THROW_ARGOSEXCEPTION("RNN Param size varies");

    }
    m_pfWeights.clear();
    for (size_t i = 0; i < un_num_params; i++){
      m_pfWeights.push_back(pf_params[i]);
    }
   }

   size_t GetLengthWeights() {
    return m_unNumberOfWeights;
   }
   
   void Reset();
   
   Real dotProd(std::vector<Real>& v1, std::vector<Real>& v2) {
     Real dotProd = 0.0f;
     for (size_t i = 0; i < v1.size(); i++) {
      dotProd += v1[i]*v2[i];
     }
     return dotProd;
   }

   void eleProd(std::vector<Real>& input, std::vector<Real>& acc) {
       for (size_t i = 0; i < input.size(); i++) {
      acc[i] *= input[i];
     }
   }

   void eleAdd(std::vector<Real>& input, std::vector<Real>& acc) {
       for (size_t i = 0; i < input.size(); i++) {
      acc[i] += input[i];
     }
   }


   void Prod(std::vector<Real>& input, std::vector<Real>& output, size_t& start, size_t outsize) {
        output.clear();
        for (size_t i = 0; i < outsize; i++) {
           std::vector<Real> temp(std::next(m_pfWeights.begin(),start), std::next(m_pfWeights.begin(), start + input.size()));
           start += input.size();
           output.push_back(dotProd(input,temp));
        }


   }

   void sigmoid(std::vector<Real>& vec) {
     for (size_t i = 0; i < vec.size(); i++) {
      vec[i] = 1.0f / ( 1.0f + std::exp( -vec[i]) );
    }

   }

   void sigmoidWithSoftMax(std::vector<Real>& vec) {
     for (size_t i = 0; i < (vec.size() - 3); i++) {
      vec[i] = 1.0f / ( 1.0f + std::exp( -vec[i]) );
    }
    
    Real max = std::max(std::max(vec[vec.size() - 3], vec[vec.size() - 2]), vec[vec.size() - 1]);
    Real e1 = std::exp(vec[vec.size() - 3] - max);
    Real e2 = std::exp(vec[vec.size() - 2] - max);
    Real e3 = std::exp(vec[vec.size() - 1] - max);
    Real sum = e1 + e2 + e3;
    vec[vec.size() - 3] = e1/sum;
    vec[vec.size() - 2] = e2/sum;
    vec[vec.size() - 1] = e3/sum;

   }




private:

   size_t  m_unInputSize, m_unOutputSize, m_unNumberOfWeights;
   size_t   m_unInputToHidden, m_unHiddenToOutput, m_unHiddenToHidden;
   std::vector<size_t>   m_unULayers, m_unVLayers, m_unWLayers;
   std::vector<Real>    m_pfWeights, m_pfOutputs, m_pfInputs;
   std::vector<std::vector<Real> >   m_pfHiddenStateVec;
   std::vector<Real> m_pfHiddenState;
  
};

#endif
