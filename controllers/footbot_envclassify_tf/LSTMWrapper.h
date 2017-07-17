#include <boost/python.hpp>
using namespace boost::python;

class LSTMWrapper {

  public:

	LSTMWrapper();
	~LSTMWrapper();
	typedef std::vector<double> IOType;
	
	IOType input_;
	IOType output_;
	void SetInput(IOType& input);
	void SetOutput(IOType& output);
	void ComputeOutput(std::string& id);
	void GetOutput(IOType& output);
    void GetInput(IOType& input);
    void SetReward(double reward, std::string& id);
    void Reset() {}
    void Destroy() {}
    template <class T> boost::python::list toPythonList(std::vector<T>& vector);
    template <class T> void fromPythonList(boost::python::list& plist, std::vector<T>& vector);

    object lstm_module, lstm_namespace, lstm_obj;

};