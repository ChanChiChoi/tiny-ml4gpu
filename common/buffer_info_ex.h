#pragma once

#include <stdio.h>
#include <vector>
#include <string>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

typedef struct buffer_info_ex: public py::buffer_info{

    void *ptr_device = nullptr; // pointer of ptr data on GPU

    buffer_info_ex():buffer_info{}{
        ptr_device = nullptr;
    }

    buffer_info_ex(buffer_info &rhs){
        ptr = rhs.ptr;
        ptr_device = nullptr;
        itemsize = rhs.itemsize;
        size = rhs.size;
        format = std::move(rhs.format);
        ndim = rhs.ndim;
        shape = std::move(rhs.shape);
        strides = std::move(rhs.strides);

    }
    
    /*
 *   just call cuda function, it will copy ptr data onto gpu
 * */
    buffer_info_ex & cuda();

    ~buffer_info_ex();
    
} Buf;

class Array{

    Buf *ptr_buf = NULL;
public:
    Array() {
        ptr_buf = new Buf();
    }
    
    Array(py::array_t<float> &array){
        auto array_info = array.request();

        if (array_info.format != py::format_descriptor<float>::format())
            throw std::runtime_error("Incompatible format: excepted a float32 array!");
        if (array_info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension! it should be 2 dim");

        ptr_buf = new Buf(array_info);
    }

    
    ~Array(){
        if (ptr_buf != NULL)
             delete ptr_buf;
             ptr_buf = NULL;
    }
    
    Array & cuda();

};
