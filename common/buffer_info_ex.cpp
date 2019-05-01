#include "common/buffer_info_ex.h"
#include "common/malloc_free.h"

Buf& buffer_info_ex::cuda(){

    size_t bytes_size = itemsize * size;
    switch (format[0]){
        case 'f':
            this -> ptr_device = host_to_device(ptr, bytes_size);
            break;
        default:
            throw std::runtime_error("current version only support float32!");
            break;
    }

    return *this;
}

Array & Array::cuda(){
    this->cuda();
    return *this;

}
