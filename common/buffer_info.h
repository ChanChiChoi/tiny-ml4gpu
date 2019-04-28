/*
 *  copy from:
    pybind11/buffer_info.h: Python buffer object interface
*/

#pragma once

#include <stdio.h>
#include <vector>
#include<string>
#include "common/common.h"


/// Information record describing a Python buffer object
typedef struct buffer_info {
    void *ptr_host = nullptr;          // Pointer to the underlying storage on host
    void *ptr_device = nullptr;        // Pointer to the underlying storage on device
    ssize_t itemsize = 0;         // Size of individual items in bytes
    ssize_t size = 0;             // Total number of entries
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
    ssize_t ndim = 0;             // Number of dimensions
    std::vector<ssize_t> shape;   // Shape of the tensor (1 entry per dimension)
    std::vector<ssize_t> strides; // Number of entries between adjacent entries (for each per dimension)

    buffer_info() {}

    buffer_info(void *ptr_host, void *ptr_device, ssize_t itemsize, Dtype dtype, ssize_t ndim,
                std::vector<ssize_t> shape, std::vector<ssize_t> strides)
    :ptr_host(ptr_host), ptr_device(ptr_device), itemsize(itemsize), dtype(dtype), ndim(ndim), 
     shape(std::move(shape)), strides(std::move(strides)){
        if (ndim != (ssize_t) shape.size() || ndim != (ssize_t) strides.size())
            assert("buffer_info: ndim do not match shape and/or strides length");
        for (size_t i = 0; i<(size_t)ndim; i++)
            size *= shape[i];
    }

//    template<typename T>
//    buffer_info(T *ptr_host, T *ptr_device, std::vector<ssize_t>shape, std::vector<ssize_t>strides)
//    :buffer_info(ptr_host, ptr_device, sizeof(T),)

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



