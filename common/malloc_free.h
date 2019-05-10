#pragma once

//=========malloc
float *
device_malloc(size_t size, const char *file, const int line);

unsigned int *
device_malloc(size_t size, const char *file, const int line);

double *
device_malloc(size_t size, const char *file, const int line);

float *
host_to_device_malloc(float * ptr_host, size_t size, const char *file, const int line);

unsigned int *
host_to_device_malloc(unsigned int * ptr_host, size_t size, const char *file, const int line);

double *
host_to_device_malloc(double * ptr_host, size_t size, const char *file, const int line);

float *
host_to_device(float *ptr_device, float * ptr_host, size_t size, const char *file, const int line);

unsigned int *
host_to_device(unsigned int *ptr_device, unsigned int * ptr_host, size_t size, const char *file, const int line);

double *
host_to_device(double *ptr_device, double * ptr_host, size_t size, const char *file, const int line);

//============marco define
#define DEVICE_MALLOC(size) device_malloc(size, __FILE__, __LINE__)
#define HOST_TO_DEVICE_MALLOC(ptr_host, size) host_to_device_malloc(ptr_host, size, __FILE__, __LINE__);
#define HOST_TO_DEVICE(ptr_device, ptr_host, size) host_to_device(ptr_device, ptr_host, size, __FILE__, __LINE__)

//===========free
float *
device_free(float *ptr_device, const char *file, const int line);

unsigned int *
device_free(unsigned int *ptr_device, const char *file, const int line);

double *
device_free(double *ptr_device, const char *file, const int line);

float *
device_to_host_free(float * ptr_host, float *ptr_device, size_t size, const char *file, const int line);

unsigned int *
device_to_host_free(unsigned int * ptr_host, unsigned int *ptr_device, size_t size, const char *file, const int line);

double *
device_to_host_free(double * ptr_host, double *ptr_device, size_t size, const char *file, const int line);

float *
device_to_host(float * ptr_host, float *ptr_device, size_t size, const char *file, const int line);

unsigned int *
device_to_host(unsigned int * ptr_host, unsigned int *ptr_device, size_t size, const char *file, const int line);

double *
device_to_host(double * ptr_host, double *ptr_device, size_t size, const char *file, const int line);

//============macro define
#define DEVICE_FREE(ptr_device) device_free(ptr_device, __FILE__, __LINE__)
#define DEVICE_TO_HOST_FREE(ptr_host, ptr_device, size) device_to_host_free(ptr_host, ptr_device, size, __FILE__, __LINE__);
#define DEVICE_TO_HOST(ptr_host, ptr_device, size) device_to_host(ptr_host, ptr_device, size, __FILE__, __LINE__)

//============change
void
device_to_device(float *dst_device, float *src_device, size_t size, const char *file, const int line);

void
device_to_device(unsigned int *dst_device, unsigned int *src_device, size_t size, const char *file, const int line);

void
device_to_device(double *dst_device, double *src_device, size_t size, const char *file, const int line);

//============macro define
#define DEVICE_TO_DEVICE(dst_device, src_device) device_to_device(dst_device, src_device, __FILE__, __LINE__)
