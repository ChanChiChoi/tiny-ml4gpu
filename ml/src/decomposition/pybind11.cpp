/*
 
g++  pybind11.cpp buffer_info_ex.cpp -shared -std=c++11 -fPIC -I../pybind11/include  -I/root/miniconda3/include/python3.6m -o Array`python3-config --extension-suffix` -I../ -L. -lmalloc_free
 * */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ml/include/decomposition/pca.h"

namespace py = pybind11;


PYBIND11_MODULE(PCA, m){

    py::class_<PCA>(m,"PCA")
        .def(py::init<int>())
        .def("fit", &PCA::fit)
        .def("transform", &PCA::transform)
        ;


}
