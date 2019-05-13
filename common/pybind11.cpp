#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "common/buffer_info_ex.h"

namespace py = pybind11;


PYBIND11_MODULE(Array, m){

    py::class_<Array>(m,"Array")
        .def(py::init<py::array_t<float> &>())
        .def("cuda", &Array::cuda)
        .def("cpu", &Array::cpu)
        ;


}
