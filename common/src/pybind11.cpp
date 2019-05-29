/*
 
g++  pybind11.cpp buffer_info_ex.cpp -shared -std=c++11 -fPIC -I../pybind11/include  -I/root/miniconda3/include/python3.6m -o Array`python3-config --extension-suffix` -I../ -L. -lmalloc_free
 * */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "common/include/buffer_info_ex.h"

namespace py = pybind11;


PYBIND11_MODULE(Array, m){

    py::class_<Array>(m,"Array")
        .def(py::init<py::array_t<float> &>())
        .def("cuda", &Array::cuda)
        .def("cpu", &Array::cpu)
        .def("display_meta", &Array::display_meta)
        .def("display_data", &Array::display_data)
        ;


}
