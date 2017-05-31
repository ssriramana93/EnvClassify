#include "rnn.h"

#include <fstream>
#include <cmath>

/****************************************/
/****************************************/

CRNN::CRNN() :
   m_unInputSize(67),
   m_unOutputSize(15),
   m_unInputToHidden(1),
   m_unHiddenToHidden(3),
   m_unHiddenToOutput(3),
   m_unNumberOfWeights(0)
    {}

/****************************************/
/****************************************/

CRNN::~CRNN() {
}

/****************************************/
/****************************************/

void CRNN::Init(TConfigurationNode& t_node) {
   /* First perform common initialisation from base class */
   GetNodeAttributeOrDefault(t_node, "num_inputs", m_unInputSize, m_unInputSize);
   GetNodeAttributeOrDefault(t_node, "num_outputs", m_unOutputSize, m_unOutputSize);

   InitNumHiddenNeurons();
   m_unNumberOfWeights = GetNumWeights();

   Reset();

//   m_pfInputs = std::vector<Real>(m_unInputSize, 0.0f);
//   InitNumHiddenNeurons();
//   InitHiddenStates();
   std::cout <<"Init Size: "<<m_pfInputs.size()<<"m_unNumberOfWeights ="<< m_unNumberOfWeights <<std::endl;
   std::cout<<"nH= "<<m_pfHiddenState.size()<<std::endl;

}

void CRNN::InitNumHiddenNeurons() {
  /*for (size_t i = 0; i <= m_unInputToHidden; i++) {
  	m_unULayers.push_back(10);
  }*/

  for (size_t i = 0; i <= m_unHiddenToHidden; i++) {
  	m_unWLayers.push_back(10);
  }

  for (size_t i = 0; i <= m_unHiddenToOutput; i++) {
  	m_unVLayers.push_back(10);
  }
}

size_t CRNN::GetNumWeights() {
	size_t size = m_unInputSize*m_unWLayers[0];
	/*for (size_t i = 1; i < m_unULayers.size(); i++) {
		size += m_unULayers[i]*m_unULayers[i-1];
	}*/
	//size += m_unVLayers[0]*m_unULayers[m_unULayers.size() - 1];
	for (size_t i = 1; i < m_unWLayers.size(); i++) {
		size += m_unWLayers[i]*m_unWLayers[i-1];
	}
    size += m_unVLayers[0]*m_unWLayers[m_unWLayers.size() - 1];
	for (size_t i = 1; i < m_unVLayers.size(); i++) {
		size += m_unVLayers[i]*m_unVLayers[i-1];
	}
	size += m_unVLayers[m_unVLayers.size() - 1]*m_unOutputSize;
	return size;
}

void CRNN::InitHiddenStates()  {
	/*for (size_t i = 1; i < m_unWLayers.size(); i++) {
		std::vector<Real> temp(m_unWLayers[i], 0.0f);
		m_pfHiddenStateVec.push_back(temp);
	}*/
	m_pfHiddenState = std::vector<Real>(m_unWLayers[0], 0.0f);
}

/****************************************/
/****************************************/


void CRNN::Reset() {
    m_pfInputs = std::vector<Real>(m_unInputSize, 0.0f);
    m_pfOutputs = std::vector<Real>(m_unOutputSize, 0.0f);
    InitHiddenStates();

	m_pfWeights.clear();
	//m_pfOutputs.clear();
	for (size_t i = 0; i < m_unNumberOfWeights; i++) {
   	m_pfWeights.push_back(0.0f);
   }
}


void CRNN::Destroy() {

  
}

/****************************************/
/****************************************/


/****************************************/
/****************************************/



/****************************************/
/****************************************/

void CRNN::ComputeOutputs() {
	size_t count = 0;
	std::vector<Real> in(m_pfInputs), out1, out2;


	Prod(m_pfInputs, out1, count, m_unWLayers[0]);

    Prod(m_pfHiddenState, out2, count, m_unWLayers[0]);


    eleAdd(out1, out2);

    sigmoid(out1);


    for (size_t i = 1; i < m_unWLayers.size(); i++) {
    	in = out1;
    	Prod(in, out1, count, m_unWLayers[i]);
  	    sigmoid(out1);


    }


    m_pfHiddenState = out1;

    for (size_t i = 1; i < m_unVLayers.size(); i++) {
    	in = out1;
    	Prod(in, out1, count, m_unVLayers[i]);
  	    sigmoid(out1);

    }

    in = out1;
    Prod(in, m_pfOutputs, count, m_unOutputSize);
    sigmoidWithSoftMax(m_pfOutputs);
  
}

/****************************************/
/****************************************/
