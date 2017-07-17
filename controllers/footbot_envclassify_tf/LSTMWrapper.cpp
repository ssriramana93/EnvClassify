#include <LSTMWrapper.h>
#include <boost/python.hpp>
#include <iostream>

	LSTMWrapper::LSTMWrapper() {
		lstm_module = import("ArgosInterfaceObject");
		lstm_namespace = lstm_module.attr("__dict__");	
		lstm_obj = lstm_module.attr("argos_interface");

	}

	LSTMWrapper::~LSTMWrapper() {
		std::cout<<"Destroy LSTM"<<std::endl;

	}

	void LSTMWrapper::SetInput(IOType& input) {
		input_ = input;
	}

	void LSTMWrapper::SetOutput(IOType& output) {
		output_ = output;
	}

	void LSTMWrapper::ComputeOutput(std::string &id) {
		list input = toPythonList<double>(input_), output;
		lstm_obj.attr("SetInput")(input, id);
		lstm_obj.attr("ComputeOutput")();
		//std::cout<<"OK Here"<<std::endl;

		output = extract<list>(lstm_obj.attr("GetOutput")());
		//std::cout<<"OK Here"<<std::endl;
		fromPythonList<double>(output, output_);

	}

	void LSTMWrapper::SetReward(double reward, std::string &id) {
		lstm_obj.attr("SetReward")(reward, id);
	}

	void LSTMWrapper::GetInput(IOType& input) {
		input = input_;
	}

	void LSTMWrapper::GetOutput(IOType& output) {
		output = output_;
	}


	template <class T> boost::python::list LSTMWrapper::toPythonList(std::vector<T>& vector) {
	typename std::vector<T>::iterator iter;
	boost::python::list list;
	for (iter = vector.begin(); iter != vector.end(); ++iter) {
		list.append(*iter);
	}
	return list;
}

template <class T> void LSTMWrapper::fromPythonList(boost::python::list& plist, std::vector<T>& vector) {
	
	vector.clear();
	for (int i = 0; i < len(plist); ++i)
    {	

        vector.push_back(boost::python::extract<T>(plist[i]));
    }
}