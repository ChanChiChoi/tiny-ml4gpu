/*
 *  copy from:
    pybind11/buffer_info.h: Python buffer object interface
*/

#pragma once

#include <stdio.h>
#include <vector>
#include<string>
#include "common/common.h"


typedef struct buffer_info {

    void *ptr_host = nullptr;          // Pointer to the underlying storage on host
    void *ptr_device = nullptr;        // Pointer to the underlying storage on device
    size_t itemsize = 0;         // Size of individual items in bytes
    size_t size = 1;             // Total number of entries
    Dtype dtype;         // record the type of item in buffer;
    std::string format;           // For compatible with pybind11 buffer_info, 

   /*
    *  ndim: 
    *    0 - scalar:  x 
    *    1 - vector[cols], [1, cols]:  [x],[x,x,x]
    *    2 - matrix[rows, cols]:   [[x,x],[x,x]]
    *
    *    ...
    *
    * */
    size_t ndim = 0;             // Number of dimensions
    std::vector<size_t> shape;   // Shape of the tensor (1 entry per dimension): must be 4 dim,[1,2,0,0]
    std::vector<size_t> strides; // Number of entries between adjacent entries (for each per dimension)

    buffer_info() {
        for (size_t i = 0; i< ndim; i++)
            size *= shape[i];
    }

    buffer_info(void * ptr_host, unsigned long itemsize, Dtype dtype, int ndim,
                any_container<size_t> shape_in , any_container<size_t> strides_in )
    :ptr_host(ptr_host), itemsize(itemsize), dtype(dtype), ndim(ndim), 
     shape(std::move(shape_in) ), strides(std::move(strides_in)){
        ptr_device = NULL;
        if (ndim != (size_t) shape.size() || ndim != (size_t) strides.size())
            assert("buffer_info: ndim do not match shape and/or strides length");
        size = 1; // in case of result of *= is 0;
        for (size_t i = 0; i<(size_t)ndim; i++)
            size *= shape[i];
    }


    buffer_info(const buffer_info &) = delete;

    buffer_info& operator=(const buffer_info &) = delete;
   
    buffer_info(buffer_info &&other){
        (*this) = std::move(other);
    }

    buffer_info& operator=(buffer_info &&rhs){
        ptr_host = rhs.ptr_host;
        ptr_device = rhs.ptr_device;
        itemsize = rhs.itemsize;
        size = rhs.size;
        ndim = rhs.ndim;
        shape = std::move(rhs.shape);
        strides = std::move(rhs.strides);
        return *this;
    }

} Buf;



