#include <string>
#include "common/malloc_free.h"
#include "common/helper.cuh"


template<class T> T *
device_malloc(size_t size){

    T *ptr_device = NULL;
    CHECK_CALL(cudaMalloc((void **)&ptr_device, size));
    return ptr_device;
}


template<typename T> T *
host_to_device(T * ptr_host, size_t size){

    T *ptr_device = device_malloc<T>(size);
    // copy host data to device;
    CHECK_CALL(cudaMemcpy(ptr_device, ptr_host, size, cudaMemcpyHostToDevice));
    return ptr_device;
}


template<typename T> T *
device_free(T *ptr_device){

    CHECK_CALL(cudaFree(ptr_device));
    return NULL;
}


template<typename T> T *
device_to_host(T * ptr_device, T *ptr_host, size_t size){

    CHECK_CALL(cudaMemcpy(ptr_host, ptr_device, size, cudaMemcpyDeviceToHost));
    device_free<T>(ptr_device);
    ptr_device = NULL;
    return ptr_device;
}
