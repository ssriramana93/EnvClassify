#include <LSTMWrapper.h>
#include <boost/python.hpp>


BOOST_PYTHON_MODULE(LSTM)
{
    class_<LSTMWrapper>("LSTMWrapper")
        .def("ComputeOutput", &World::ComputeOutput)
        .def("ComputeOutput", &World::ComputeOutput)
    ;
}