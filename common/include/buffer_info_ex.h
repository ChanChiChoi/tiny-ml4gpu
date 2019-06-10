#pragma once

#include <stdio.h>
#include <vector>
#include <string>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

typedef struct buffer_info_ex: public py::buffer_info{

    // malloc from GPU;
    void *ptr_device = nullptr; 
    // malloc from cpu; another ptr just fro receive from python numpy
    void *ptr_host = nullptr; 
    
    
    buffer_info_ex():buffer_info{}{
        ptr_device = nullptr;
        ptr_host = nullptr;
    }

    buffer_info_ex(buffer_info &rhs){
        ptr = rhs.ptr;
        ptr_device = nullptr;
        ptr_host = nullptr;
        itemsize = rhs.itemsize;
        size = rhs.size;
        format = std::move(rhs.format); // const char c = "?bBhHiIqQfdg"[detail::is_fmt_numeric<T>::index]
        ndim = rhs.ndim;
        shape = std::move(rhs.shape);
        strides = std::move(rhs.strides);

    }
    
    /*
 *   just call cuda function, it will copy ptr data onto gpu
 * */
    buffer_info_ex & cuda();
    buffer_info_ex & cpu();
    ~buffer_info_ex();
    
} Buf;

class Array{

public:
    Buf *ptr_buf = NULL;
public:
    Array() {
        ptr_buf = new Buf();
    }
    
    Array(ssize_t rows, ssize_t cols, const std::string &format){
        ptr_buf = new Buf();
        ptr_buf->shape = std::move(std::vector<ssize_t> {rows, cols});
        ptr_buf->format = std::move(format);
    }

   Array(void *ptr, void *ptr_host, void *ptr_device, 
         const ssize_t ndim, const std::vector<ssize_t> shape, const std::string &format,
         const ssize_t itemsize, const ssize_t size, const std::vector<ssize_t> strides){
        ptr_buf = new Buf();
        ptr_buf->ptr = ptr;
        ptr_buf->ptr_host = ptr_host;
        ptr_buf->ptr_device = ptr_device;
        ptr_buf->ndim = ndim;
        ptr_buf->shape = std::move(shape);
        ptr_buf->format = std::move(format);
        ptr_buf->itemsize = itemsize;
        ptr_buf->size = size;
        ptr_buf->strides = std::move(strides);

    }

    Array(py::array_t<int> &array){
        auto array_info = array.request();

        if (array_info.format != py::format_descriptor<int>::format())
            throw std::runtime_error("Incompatible format: excepted a int32 array!");
        if (array_info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension! it should be 2 dim");

        ptr_buf = new Buf(array_info);
    }


    Array(py::array_t<float> &array){
        auto array_info = array.request();

        if (array_info.format != py::format_descriptor<float>::format())
            throw std::runtime_error("Incompatible format: excepted a float32 array!");
        if (array_info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension! it should be 2 dim");

        ptr_buf = new Buf(array_info);
    }

    Array(py::array_t<double> &array){
        auto array_info = array.request();

        if (array_info.format != py::format_descriptor<double>::format())
            throw std::runtime_error("Incompatible format: excepted a float64 array!");
        if (array_info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension! it should be 2 dim");

        ptr_buf = new Buf(array_info);
    }

    
    ~Array(){
        if (ptr_buf != NULL)
             delete ptr_buf;
             ptr_buf = NULL;
    }
    
    Array & cuda(); // transfer ptr->data into GPU
    template<typename T> py::array_t<T>  _cpu(); // transfer ptr->data from GPU into python numpy

    py::array_t<float>  cpu();
    void display_meta(); //  display metainfo of Array
    void display_cpu(); // display data of ptr_host or ptr

    void display_cuda();// display data from cuda side

};
