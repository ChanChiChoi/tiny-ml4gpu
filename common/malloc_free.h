#pragma once

template<class T> T *
device_malloc(size_t size);

template<typename T> T *
host_to_device(T * ptr_host, size_t size);

template<typename T> T *
device_free(T *ptr_device);

template<typename T> T *
device_to_host(T * ptr_device, T *ptr_host, size_t size);
