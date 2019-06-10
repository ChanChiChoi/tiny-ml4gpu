/*
 
g++  pybind11.cpp buffer_info_ex.cpp -shared -std=c++11 -fPIC -I../pybind11/include  -I/root/miniconda3/include/python3.6m -o Array`python3-config --extension-suffix` -I../ -L. -lmalloc_free
 * */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "common/include/buffer_info_ex.h"

namespace py = pybind11;


PYBIND11_MODULE(Array, m){

    py::class_<Array>(m,"Array")
        .def(py::init<py::array_t<int> &>())
        .def(py::init<py::array_t<float> &>())
        .def(py::init<py::array_t<double> &>())
        .def("cuda", &Array::cuda)
        .def("cpu", (py::array_t<int> (Array::*)(int)) &Array::cpu,"return int", py::arg("_i")=1)
        .def("cpu", (py::array_t<float> (Array::*)(float)) &Array::cpu,"return float", py::arg("_f")=1.0)
        .def("cpu", (py::array_t<double> (Array::*)(double)) &Array::cpu,"return double", py::arg("_d")=1.0)
        .def("display_meta", &Array::display_meta)
        .def("display_cpu", &Array::display_cpu)
        .def("display_cuda", &Array::display_cuda)
        ;


}
