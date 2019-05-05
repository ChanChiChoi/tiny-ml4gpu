#include <string>
#include "common/malloc_free.h"
#include "common/helper.cuh"
/*
 because nvcc will not Instantiate template onto host side of shared object file,
so, other host shared object file by g++  can not find function symbols in shared object by nvcc
*/

//==============malloc template
template<class T> T *
device_malloc(size_t size){

    T *ptr_device = NULL;
    CHECK_CALL(cudaMalloc((void **)&ptr_device, size));
    return ptr_device;
}

template<typename T> T *
_host_to_device_malloc(T *ptr_host, size_t size){

    T *ptr_device = device_malloc<T>(size);
    // copy host data to device;
    CHECK_CALL(cudaMemcpy(ptr_device, ptr_host, size, cudaMemcpyHostToDevice));
    return ptr_device;
}

template<typename T> T *
_host_to_device(T *ptr_device, T *ptr_host, size_t size){

    // copy host data to device;
    CHECK_CALL(cudaMemcpy(ptr_device, ptr_host, size, cudaMemcpyHostToDevice));
    return ptr_device;
}

//===========template Instantiation
float *
host_to_device_malloc(float *ptr_host, size_t size){
    return _host_to_device_malloc<float>(ptr_host, size);
}

unsigned int *
host_to_device_malloc(unsigned int * ptr_host, size_t size){
    return _host_to_device_malloc<unsigned int>(ptr_host, size);
}

float *
host_to_device(float *ptr_device, float *ptr_host, size_t size){
    return _host_to_device<float>(ptr_device, ptr_host, size);
}

unsigned int *
host_to_device(unsigned int *ptr_device, unsigned int *ptr_host, size_t size){
    return _host_to_device<unsigned int>(ptr_device, ptr_host, size);
}

//==================free template
template<typename T> T *
device_free(T *ptr_device){

    CHECK_CALL(cudaFree(ptr_device));
    return NULL;
}


template <typename T> T*
_device_to_host_free(T * ptr_host, T *ptr_device, size_t size){

    CHECK_CALL(cudaMemcpy(ptr_host, ptr_device, size, cudaMemcpyDeviceToHost));
    device_free<T>(ptr_device);
    ptr_device = NULL;
    return ptr_device;
}

template <typename T> T*
_device_to_host(T * ptr_host, T *ptr_device, size_t size){

    CHECK_CALL(cudaMemcpy(ptr_host, ptr_device, size, cudaMemcpyDeviceToHost));
    return ptr_device;
}

//=============template Instantiation
float *
device_to_host_free(float * ptr_host, float *ptr_device, size_t size){
    return _device_to_host_free<float>(ptr_host, ptr_device, size);
}

unsigned int *
device_to_host_free(unsigned int * ptr_host, unsigned int  *ptr_device, size_t size){
    return _device_to_host_free<unsigned int>(ptr_host, ptr_device, size);
}

float *
device_to_host(float * ptr_host, float *ptr_device, size_t size){
    return _device_to_host<float>(ptr_host, ptr_device, size);
}

unsigned int *
device_to_host(unsigned int * ptr_host, unsigned int  *ptr_device, size_t size){
    return _device_to_host<unsigned int>(ptr_host, ptr_device, size);
}

