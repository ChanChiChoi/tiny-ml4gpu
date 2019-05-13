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
            if(ptr_device){
                DEVICE_FREE((float *)ptr_device);
                ptr_device = NULL;
            }
            if(ptr_host){
                free((float *)ptr_host);
                ptr_host = NULL;
            }
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

template<typename T> py::array_t<T> &
Array::cpu(){
    
    assert(this->ptr_buf);

    void * res = nullptr;
    if (ptr_buf -> ptr_host){
        res = ptr_buf->ptr_host
    }else if{
        res = ptr_buf->ptr
    }else{
        res = nullptr;
        throw std::runtime_error("current has no data");
    }
   
    auto res =  py::buffer_info(
                res,
                ptr_buf->itemsize,
                ptr_buf->format,
                ptr_buf->ndim,
                ptr_buf->shape,
                ptr_buf->strides
           );

    return py::array_t<T>{*res};

}
