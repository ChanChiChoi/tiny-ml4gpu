#pragma once

//=========malloc
template<class T> T *
device_malloc(size_t size);

float *
host_to_device_malloc(float * ptr_host, size_t size);

unsigned int *
host_to_device_malloc(unsigned int * ptr_host, size_t size);

float *
host_to_device(float * ptr_host, size_t size);

unsigned int *
host_to_device(unsigned int * ptr_host, size_t size);

//===========free
template<typename T> T *
device_free(T *ptr_device);

float *
device_to_host_free(float * ptr_device, float *ptr_host, size_t size);

unsigned int *
device_to_host_free(unsigned int * ptr_device, unsigned int *ptr_host, size_t size);

float *
device_to_host(float * ptr_device, float *ptr_host, size_t size);

unsigned int *
device_to_host(unsigned int * ptr_device, unsigned int *ptr_host, size_t size);
