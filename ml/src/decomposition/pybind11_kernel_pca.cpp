/*
 
g++  pybind11.cpp buffer_info_ex.cpp -shared -std=c++11 -fPIC -I../pybind11/include  -I/root/miniconda3/include/python3.6m -o Array`python3-config --extension-suffix` -I../ -L. -lmalloc_free
 * */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ml/include/decomposition/kernel_pca.h"

namespace py = pybind11;


PYBIND11_MODULE(KPCA, m){

    py::class_<KPCA>(m,"KPCA")
        .def(py::init<int>())
        .def("fit", &KPCA::fit)
        .def("transform", &KPCA::transform)
        ;


}
