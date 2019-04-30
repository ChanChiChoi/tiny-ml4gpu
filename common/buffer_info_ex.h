#pragma once

#include <stdio.h>
#include <vector>
#include <string>
#include "pybind11/buffer_info.h"
#include "pybind11/detail/common.h"

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


} Buf;
