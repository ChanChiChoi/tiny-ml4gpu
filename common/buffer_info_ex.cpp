#include "common/buffer_info_ex.h"
#include "common/malloc_free.h"

Buf& buffer_info_ex::cuda(){

    size_t bytes_size = itemsize * size;
    switch (format[0]){
        case 'f':
            ptr_device = (void *)HOST_TO_DEVICE_MALLOC((float *)ptr, bytes_size);
            break;
        default:
            throw std::runtime_error("current version only support float32!");
            break;
    }

    return *this;
}

buffer_info_ex::~buffer_info_ex(){

    switch (format[0]){
        case 'f':
            DEVICE_FREE((float *)ptr_device);
            ptr_device = NULL;
            break;
        default:
            throw std::runtime_error("current version only support float32!");
            break;
    }

}


Array & Array::cuda(){
    ptr_buf->cuda();
    return *this;

}
