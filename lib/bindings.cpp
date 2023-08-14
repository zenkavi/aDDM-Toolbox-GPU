#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/gpu_toolbox.h"

namespace py = pybind11; 

using namespace pybind11::literals; 

PYBIND11_MODULE(addm_toolbox_gpu, m) {
    m.doc() = "aDDMToolbox for the GPU.";
    py::class_<ProbabilityData>(m, "ProbabilityData")
        .def(py::init<double, double>(), 
            py::arg("likelihood")=0, 
            py::arg("NLL")=0)
        .def_readonly("likelihood", &ProbabilityData::likelihood)
        .def_readonly("NLL", &ProbabilityData::NLL)
        .def_readonly("trialLikelihoods", &ProbabilityData::trialLikelihoods);
    py::class_<DDMTrial>(m, "DDMTrial")
        .def(py::init<int, int, int, int>(), "Create a DDMTrial")
        .def_readonly("RT", &DDMTrial::RT)
        .def_readonly("choice", &DDMTrial::choice)
        .def_readonly("valueLeft", &DDMTrial::valueLeft)
        .def_readonly("valueRight", &DDMTrial::valueRight)
        .def_readonly("RDVs", &DDMTrial::RDVs)
        .def_readonly("timeStep", &DDMTrial::timeStep)
        .def_static("writeTrialsToCSV", &DDMTrial::writeTrialsToCSV)
        .def_static("loadTrialsFromCSV", &DDMTrial::loadTrialsFromCSV);
    py::class_<DDM>(m, "DDM")
        .def(py::init<float, float, float, unsigned int, float, float>(), 
            py::arg("d"), 
            py::arg("sigma"), 
            py::arg("barrier")=1, 
            py::arg("nonDecisionTime")=0, 
            py::arg("bias")=0, 
            py::arg("decay")=0)
        .def_readonly("d", &DDM::d)
        .def_readonly("sigma", &DDM::sigma)
        .def_readonly("barrier", &DDM::barrier)
        .def_readonly("nonDecisionTime", &DDM::nonDecisionTime)
        .def_readonly("bias", &DDM::bias)
        .def_readonly("decay", &DDM::decay)
        .def("simulateTrial", &DDM::simulateTrial, 
            py::arg("valueLeft"), 
            py::arg("valueRight"),
            py::arg("timeStep")=1, 
            py::arg("seed")=-1);
}
