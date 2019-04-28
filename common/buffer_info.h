/*
 *  copy from:
    pybind11/buffer_info.h: Python buffer object interface
*/

#pragma once

#include <stdio.h>
#include <vector>
#include<string>
//#include "common/common.h"


/// Information record describing a Python buffer object
typedef struct buffer_info {
    void *ptr_host = nullptr;          // Pointer to the underlying storage on host
    void *ptr_device = nullptr;        // Pointer to the underlying storage on device
    ssize_t itemsize = 0;         // Size of individual items in bytes
    ssize_t size = 0;             // Total number of entries
    std::string itemtype;         // record the type of item in buffer;
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

} Buf;



