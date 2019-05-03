#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include<stdio.h>

#include "sort.h"

namespace py = pybind11;

void sort(py::array_t<unsigned int> &array, py::array_t<unsigned int> &ind){

    py::buffer_info buf1 = array.request();
    py::buffer_info buf2 = ind.request();

    int rows = buf1.shape[0];//30
    int cols = buf1.shape[1];//200000
    sort_by_rows_cpu((unsigned int *)buf1.ptr, (unsigned int *)buf2.ptr,rows,cols);
    printf("rows %d cols %d\n",rows, cols);
    printf("sizeof %ld\n",sizeof(unsigned int));

}



PYBIND11_MODULE(example,m){

    m.def("sort",&sort);

}


