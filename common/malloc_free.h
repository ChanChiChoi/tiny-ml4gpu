
#include<cuda_runtime.h>

#ifndef __MALLOC_FREE__
#define __MALLOC_FREE__
#include "common/buffer_info_ex.h"

template<class T> T * device_malloc(size_t size);

void host_to_device(Buf &buf);

template <class T> void device_free(T *pdevice);
void device_to_host(Buf &buf);

#endif
