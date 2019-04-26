
#include "common/malloc_free.h"
#include "common/helper.h"
#include "common/common.h"



template<class T> T *
device_malloc(size_t size){
    
    T *pdevice = NULL;
    CHECK_CALL(cudaMalloc((void **)&pdevice, size)):
    return pdevice;

}


template<class T> T * 
host_to_device(T *){
    
    T *pdevice = device_malloc<T>();

}
 
/* */

