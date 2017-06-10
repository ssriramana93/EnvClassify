#include <boost/python.hpp>

class LSTMWrapper {
	LSTMWrapper();
	~LSTMWrapper();
	typedef std::vector<Real> InputType;
	typedef std::vector<Real> OutputType;
	void SetInput(InputType& input);
	void SetOutput(OutputType& output);
	void ComputeOutput();
	void GetOutput(OutputType& output);
    void GetInput(InputType& input);

}