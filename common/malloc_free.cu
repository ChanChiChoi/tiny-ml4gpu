#include <string>
#include "common/malloc_free.h"
#include "common/helper.cuh"
/*
 because nvcc will not Instantiate template onto host side of shared object file,
so, other host shared object file by g++  can not find function symbols in shared object by nvcc
*/

template<class T> T *
device_malloc(size_t size){

    T *ptr_device = NULL;
    CHECK_CALL(cudaMalloc((void **)&ptr_device, size));
    return ptr_device;
}


float *
host_to_device(float * ptr_host, size_t size){

    float *ptr_device = device_malloc<float>(size);
    // copy host data to device;
    CHECK_CALL(cudaMemcpy(ptr_device, ptr_host, size, cudaMemcpyHostToDevice));
    return ptr_device;
}


template<typename T> T *
device_free(T *ptr_device){

    CHECK_CALL(cudaFree(ptr_device));
    return NULL;
}


float *
device_to_host(float * ptr_device, float *ptr_host, size_t size){

    CHECK_CALL(cudaMemcpy(ptr_host, ptr_device, size, cudaMemcpyDeviceToHost));
    device_free<float>(ptr_device);
    ptr_device = NULL;
    return ptr_device;
}
