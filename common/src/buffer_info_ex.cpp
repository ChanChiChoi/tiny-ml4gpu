#include <stdio.h>
#include "common/include/buffer_info_ex.h"
#include "common/include/malloc_free.h"

Buf& buffer_info_ex::cuda(){

    size_t bytes_size = itemsize * size;
    switch (format[0]){
        case 'f':
            ptr_device = (void *)HOST_TO_DEVICE_MALLOC((float *)ptr, bytes_size);
            break;
        case NULL:
           printf("current Array obj has no data concent, cannot execute cuda()!\n");
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
        case NULL:
           printf("current Array obj has no data concent, cannot deconstructor!\n");
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

template<typename T> py::array_t<T> 
Array::_cpu(){
    
    assert(this->ptr_buf);

    void * res = nullptr;
    if (ptr_buf -> ptr_host){
        res = ptr_buf->ptr_host;
    }else if(ptr_buf->ptr){
        res = ptr_buf->ptr;
    }else{
        res = nullptr;
        throw std::runtime_error("current has no data");
    }
   
    py::buffer_info res_buf =  py::buffer_info(
                res,
                ptr_buf->itemsize,
                ptr_buf->format,
                ptr_buf->ndim,
                ptr_buf->shape,
                ptr_buf->strides
           );

    return py::array_t<T>{res_buf};

}

py::array_t<float> 
Array::cpu(){

    return this->_cpu<float>();
}

void
Array::display_meta(){
    printf("ptr id: %ld\n",ptr_buf->ptr);
    printf("ptr_host id: %d\n",ptr_buf->ptr_host);
    printf("ptr_device id: %d\n",ptr_buf->ptr_device);
    printf("ndim: %d\n",ptr_buf->ndim);
    printf("format: %s\n",ptr_buf->format.c_str());
    printf("itemsize: %d\n",ptr_buf->itemsize);
    printf("size: %d\n",ptr_buf->size);
}

void
Array::display_data(){
        
     if(this->ptr_buf->ptr_host){
         auto pdata = this->ptr_buf->ptr_host;
     }else if(this->ptr_buf->ptr){
         auto pdata = this->ptr_buf->ptr;
     }else{
         float *pdata = nullptr;
         throw std::runtime_error("current has no data or data only on device, not on host!");
     }

    if (this->ptr_buf->ndim == 2){
        for(size_t i=0; i<this->ptr_buf->shape[0]; i++){
            printf("\n[%d] ",i);
            for(size_t j=0; j<this->ptr_buf->shape[1]; j++){
                printf(" %lf",(double)pdata[i*ptr_buf->shape[1]+j]);
            }
        }
    }else if(this->ptr_buf->ndim == 1){
       for(size_t i=0; i<this->ptr_buf->shape[0]; i++)
            printf(" %lf",(double)pdata[i]);
    }

}  
