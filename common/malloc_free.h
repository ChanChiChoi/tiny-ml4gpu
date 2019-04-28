
#include<cuda_runtime.h>

#ifndef __MALLOC_FREE__
#define __MALLOC_FREE__
#include "common/buffer_info.h"

void host_to_device(Buf buf);

void device_to_host(Buf buf);

#endif
