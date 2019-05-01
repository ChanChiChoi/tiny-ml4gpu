#include "common/buffer_info_ex.h"
#include "common/malloc_free.h"

Buf& buffer_info_ex::cuda(){

    size_t bytes_size = itemsize * size;
    switch (format[0]){
        case 'f':
            ptr_device = (void *)host_to_device((float *)ptr, bytes_size);
            break;
        default:
            throw std::runtime_error("current version only support float32!");
            break;
    }

    return *this;
}

Array & Array::cuda(){
    ptr_buf->cuda();
    return *this;

}
