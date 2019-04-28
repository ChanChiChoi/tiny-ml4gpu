#include <typeinfo.h>
#include "common/malloc_free.h"
#include "common/helper.h"
#include "common/common.h"
#include "common/buffer_info.h"


template<class T> T *
device_malloc(size_t size){
    
    T *pdevice = NULL;
    CHECK_CALL(cudaMalloc((void **)&pdevice, size)):
    return pdevice;
}


void 
host_to_device(Buf buf){
    
    size_t size = buf.itemsize * buf.size; 

    // call device_malloc for malloc buffer on device;
    switch (buf.type){
        case "int":
            size = sizeof(int)*size;
            buf.ptr_device = device_malloc<int>(size);
            break;

        case "float":
            size = sizeof(float)*size;
            buf.ptr_device = device_malloc<float>(size): 
            break;

        default:
            assert("current version not support other types, except int and float!" == 0);
            break;
    }

    // copy host data to device;

    CHECK_CALL(cudaMemcpy(buf.ptr_device, buf.ptr_host, size, cudaMemcpyHostToDevice));
   
}
 
/* */

void
device_free(Buf buf){

    CHECK_CALL(cudaFree(buf.ptr_device));
}

void
device_to_host(Buf buf){
    size_t = buf.itemsize * buf.size;

    switch (buf.type){
        case "int":
            size = sizeof(int)*size;
            break;
        case "float":
            size = sizeof(float)*size;
            break;
        default:
            assert("current version not support other types, except int and float!" == 0);
            break;
    }

    CHECK_CALL(cudaMemcpy(buf.ptr_host, buf.ptr_device, size, cudaMemcpyDeviceToHost));

    device_free(buf);

}
