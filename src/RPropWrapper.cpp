#include <boost/python.hpp>
#include "RPropNetwork.h"
#include "FFNetwork.h"

BOOST_PYTHON_MODULE(ann)
{
    using namespace boost::python;
    // Base network class
    class_<FFNetwork>("FFNetwork", init<unsigned int, unsigned int, unsigned int>())
            .add_property("numOfInputs", &FFNetwork::getNumOfInputs)
            .add_property("numOfHidden", &FFNetwork::getNumOfHidden)
            .add_property("numOfOutputs", &FFNetwork::getNumOfOutputs)
            //.def("output", &FFNetwork::output)
        ;

    // RPropNetwork inherits from FFNetwork


    //def("getRPropNetwork", getRPropNetwork);
}
