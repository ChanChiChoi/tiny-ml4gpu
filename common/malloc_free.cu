#include <typeinfo>
#include "common/malloc_free.h"
#include "common/helper.h"
#include "common/common.h"
#include "common/buffer_info.h"



template<class T> T *
device_malloc(size_t size){
    
    T *pdevice = NULL;
    CHECK_CALL(cudaMalloc((void **)&pdevice, size));
    return pdevice;
}


void 
host_to_device(Buf &buf){
    
    size_t size = buf.itemsize * buf.size; 

    // call device_malloc for malloc buffer on device;
    switch (buf.dtype){
        case Dtype::int:
            buf.ptr_device = device_malloc<int>(size);
            break;

        case Dtype::float:
            buf.ptr_device = device_malloc<float>(size);
            break;

        default:
            assert("current version not support other types, except int and float!" == 0);
            break;
    }

    // copy host data to device;

    CHECK_CALL(cudaMemcpy(buf.ptr_device, buf.ptr_host, size, cudaMemcpyHostToDevice));
   
}
 
/* */

template <class T> void
device_free(T *pdevice){

    CHECK_CALL(cudaFree(pdevice));
}

void
device_to_host(Buf &buf){
    size_t size = buf.itemsize * buf.size;

    CHECK_CALL(cudaMemcpy(buf.ptr_host, buf.ptr_device, size, cudaMemcpyDeviceToHost));

    device_free(buf);
    buf.ptr_device = NULL;
}
