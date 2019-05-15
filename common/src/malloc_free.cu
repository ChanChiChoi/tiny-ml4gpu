#include <string>
#include "common/malloc_free.h"
#include "common/helper.cuh"
/*
 because nvcc will not Instantiate template onto host side of shared object file,
so, other host shared object file by g++  can not find function symbols in shared object by nvcc
*/

//==============malloc template
template<class T> T *
_device_malloc(T *&ptr_device, size_t size, const char *file, const int line){

    CHECK_CALL(cudaMalloc((void **)&ptr_device, size), file, line);
    return ptr_device;
}

template<typename T> T *
_host_to_device_malloc(T *ptr_host, size_t size, const char *file, const int line){

    T *ptr_device = NULL;
    ptr_device = _device_malloc<T>( ptr_device, size, file, line);
    // copy host data to device;
    CHECK_CALL(cudaMemcpy(ptr_device, ptr_host, size, cudaMemcpyHostToDevice), file, line);
    return ptr_device;
}

template<typename T> T *
_host_to_device(T *ptr_device, T *ptr_host, size_t size, const char *file, const int line){

    // copy host data to device;
    CHECK_CALL(cudaMemcpy(ptr_device, ptr_host, size, cudaMemcpyHostToDevice), file ,line);
    return ptr_device;
}

//===========template Instantiation
float *
device_malloc(float *&ptr_device, size_t size, const char *file, const int line){
    return _device_malloc<float>(ptr_device, size, file, line);
}

unsigned int *
device_malloc(unsigned int *&ptr_device, size_t size, const char *file, const int line){
    return _device_malloc<unsigned int>(ptr_device, size, file, line);
}

int *
device_malloc(int *&ptr_device, size_t size, const char *file, const int line){
    return _device_malloc<int>(ptr_device, size, file, line);
}

double *
device_malloc(double *&ptr_device, size_t size, const char *file, const int line){
    return _device_malloc<double>(ptr_device, size, file, line);
}

float *
host_to_device_malloc(float *ptr_host, size_t size, const char *file, const int line){
    return _host_to_device_malloc<float>(ptr_host, size, file, line);
}

unsigned int *
host_to_device_malloc(unsigned int * ptr_host, size_t size, const char *file, const int line){
    return _host_to_device_malloc<unsigned int>(ptr_host, size, file, line);
}

int *
host_to_device_malloc(int * ptr_host, size_t size, const char *file, const int line){
    return _host_to_device_malloc<int>(ptr_host, size, file, line);
}

double *
host_to_device_malloc(double * ptr_host, size_t size, const char *file, const int line){
    return _host_to_device_malloc<double>(ptr_host, size, file, line);
}

float *
host_to_device(float *ptr_device, float *ptr_host, size_t size, const char *file, const int line){
    return _host_to_device<float>(ptr_device, ptr_host, size, file, line);
}

unsigned int *
host_to_device(unsigned int *ptr_device, unsigned int *ptr_host, size_t size, const char *file, const int line){
    return _host_to_device<unsigned int>(ptr_device, ptr_host, size, file, line);
}

int *
host_to_device(int *ptr_device, int *ptr_host, size_t size, const char *file, const int line){
    return _host_to_device<int>(ptr_device, ptr_host, size, file, line);
}

double *
host_to_device(double *ptr_device, double *ptr_host, size_t size, const char *file, const int line){
    return _host_to_device<double>(ptr_device, ptr_host, size, file, line);
}

//==================free template
template<typename T> T *
_device_free(T *ptr_device, const char *file, const int line){

    CHECK_CALL(cudaFree(ptr_device), file, line);
    return NULL;
}


template <typename T> T*
_device_to_host_free(T * ptr_host, T *ptr_device, size_t size, const char *file, const int line){

    CHECK_CALL(cudaMemcpy(ptr_host, ptr_device, size, cudaMemcpyDeviceToHost), file, line);
    _device_free<T>(ptr_device, file, line);
    ptr_device = NULL;
    return ptr_device;
}

template <typename T> T*
_device_to_host(T * ptr_host, T *ptr_device, size_t size, const char *file, const int line){

    CHECK_CALL(cudaMemcpy(ptr_host, ptr_device, size, cudaMemcpyDeviceToHost), file, line);
    return ptr_device;
}

//=============template Instantiation
float *
device_free(float *ptr_device, const char *file, const int line){
    return _device_free<float>(ptr_device, file, line);
}

unsigned int *
device_free(unsigned int *ptr_device, const char *file, const int line){
    return _device_free<unsigned int>(ptr_device, file, line);
}

int *
device_free(int *ptr_device, const char *file, const int line){
    return _device_free<int>(ptr_device, file, line);
}

double *
device_free(double *ptr_device, const char *file, const int line){
    return _device_free<double>(ptr_device, file, line);
}

float *
device_to_host_free(float * ptr_host, float *ptr_device, size_t size, const char *file, const int line){
    return _device_to_host_free<float>(ptr_host, ptr_device, size, file, line);
}

unsigned int *
device_to_host_free(unsigned int * ptr_host, unsigned int  *ptr_device, size_t size, const char *file, const int line){
    return _device_to_host_free<unsigned int>(ptr_host, ptr_device, size, file, line);
}

int *
device_to_host_free(int * ptr_host, int  *ptr_device, size_t size, const char *file, const int line){
    return _device_to_host_free<int>(ptr_host, ptr_device, size, file, line);
}

double *
device_to_host_free(double * ptr_host, double  *ptr_device, size_t size, const char *file, const int line){
    return _device_to_host_free<double>(ptr_host, ptr_device, size, file, line);
}

float *
device_to_host(float * ptr_host, float *ptr_device, size_t size, const char *file, const int line){
    return _device_to_host<float>(ptr_host, ptr_device, size, file, line);
}

unsigned int *
device_to_host(unsigned int * ptr_host, unsigned int  *ptr_device, size_t size, const char *file, const int line){
    return _device_to_host<unsigned int>(ptr_host, ptr_device, size, file, line);
}

int *
device_to_host(int * ptr_host, int  *ptr_device, size_t size, const char *file, const int line){
    return _device_to_host<int>(ptr_host, ptr_device, size, file, line);
}

double *
device_to_host(double * ptr_host, double  *ptr_device, size_t size, const char *file, const int line){
    return _device_to_host<double>(ptr_host, ptr_device, size, file, line);
}

//=====================change template
template<typename T> void
_device_to_device(T *dst_device, T *src_device, size_t size, const char *file, const int line){
    CHECK_CALL(cudaMemcpy(dst_device, src_device, size, cudaMemcpyDeviceToDevice), file, line);
}

//===============change template Instantiation
void
device_to_device(float *dst_device, float *src_device, size_t size, const char *file, const int line){
     _device_to_device<float>(dst_device, src_device, size, file, line);
}

void 
device_to_device(unsigned int *dst_device, unsigned int *src_device, size_t size, const char *file, const int line){
     _device_to_device<unsigned int>(dst_device, src_device, size, file, line);
}

void 
device_to_device(int *dst_device, int *src_device, size_t size, const char *file, const int line){
     _device_to_device<int>(dst_device, src_device, size, file, line);
}

void 
device_to_device(double *dst_device, double *src_device, size_t size, const char *file, const int line){
     _device_to_device<double>(dst_device, src_device, size, file, line);
}

//===============cusolver

