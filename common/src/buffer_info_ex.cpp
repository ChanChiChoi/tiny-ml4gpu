#include <stdio.h>
#include <vector>
#include "common/include/buffer_info_ex.h"
#include "common/include/malloc_free.h"

Buf& buffer_info_ex::cuda(){

    size_t bytes_size = itemsize * size;
    switch (format[0]){
        case 'i':
            ptr_device = (void *)HOST_TO_DEVICE_MALLOC((int *)ptr, bytes_size);
            break;
        case 'f':
            ptr_device = (void *)HOST_TO_DEVICE_MALLOC((float *)ptr, bytes_size);
            break;
        case 'd':
            ptr_device = (void *)HOST_TO_DEVICE_MALLOC((double *)ptr, bytes_size);
            break;
        case NULL:
           printf("current Array obj has no data concent, cannot execute cuda()!\n");
           break;
        default:
            throw std::runtime_error("current version only support int32;float32;float64!");
            break;
    }

    return *this;
}

buffer_info_ex::~buffer_info_ex(){

    switch (format[0]){
        case 'i':
            if(ptr_device){
                DEVICE_FREE((int *)ptr_device);
                ptr_device = NULL;
            }
            if(ptr_host){
                free((int *)ptr_host);
                ptr_host = NULL;
            }
            break;
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
        case 'd':
            if(ptr_device){
                DEVICE_FREE((double *)ptr_device);
                ptr_device = NULL;
            }
            if(ptr_host){
                free((double *)ptr_host);
                ptr_host = NULL;
            }
            break;
        case NULL:
           printf("current Array obj has no data concent, cannot deconstructor!\n");
           break;
        default:
            throw std::runtime_error("current version only support int32;float32;float64!");
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

    void *res = nullptr;
    if (ptr_buf -> ptr_host){
        res = ptr_buf->ptr_host;
    }else if(ptr_buf->ptr){
        res = ptr_buf->ptr;
    }else{
        res = nullptr;
        throw std::runtime_error("current host side has no data");
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

py::array_t<int> 
Array::cpu(int _i){
    
    return this->_cpu<int>();
}

py::array_t<float> 
Array::cpu(float _f){
    
    return this->_cpu<float>();
}

py::array_t<double> 
Array::cpu(double _d){
    
    return this->_cpu<double>();
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

template<typename T> void
_display(T *pdata, ssize_t ndim, std::vector<ssize_t> &shape){

    if (ndim == 2){
        auto rows = shape[0];
        auto cols = shape[1];
        for(size_t i=0; i<rows; i++){
            printf("\n[%d] ",i);
            for(size_t j=0; j<cols; j++){
                printf(" %lf",*( pdata + i*cols + j) );
            }
        }
        printf("\n");
    }else if(ndim == 1){
       auto rows = shape[0];
       for(size_t i=0; i<rows; i++)
            printf(" %lf",*( pdata + i) );
       printf("\n");
    }else{
        throw std::runtime_error("current only support ndim == 1 or 2!");
    }


}

void
Array::display_cpu(){
        
     // display values on ptr_host or ptr; the result should equal dislay_cuda()
     void *_pdata = nullptr;
     if(this->ptr_buf->ptr_host){
         _pdata = this->ptr_buf->ptr_host;
     }else if(this->ptr_buf->ptr){
         _pdata = this->ptr_buf->ptr;
     }else{
         _pdata = nullptr;
         throw std::runtime_error("current has no data, or data only on device, not on host!");
     }

     
    switch (ptr_buf->format[0]){
        case 'i':
            _display((int *)_pdata, ptr_buf->ndim, ptr_buf->shape);
            break;
        case 'f':
            _display((float *)_pdata, ptr_buf->ndim, ptr_buf->shape);
            break;
        case 'd':
            _display((double *)_pdata, ptr_buf->ndim, ptr_buf->shape);
            break;
        case NULL:
           printf("current Array obj has no data concent, cannot execute display_data()!\n");
           break;
        default:
            throw std::runtime_error("current version only support int32;float32;float64!");
            break;
    }
}  

void
Array::display_cuda(){
    
    // copy data from device side to host side, then print them on host side!,the result should equal display_cpu()
    void *_pdata_device = nullptr;
    if(this->ptr_buf->ptr_device){
        _pdata_device = this->ptr_buf->ptr_device;

    }else{
        throw std::runtime_error("current has no data on device!");
    }
    
    switch (ptr_buf->format[0]){
        case 'i':{
            ssize_t itemsize = ptr_buf->itemsize;
            ssize_t size = ptr_buf->size;
            int *_pdata = (int *)malloc(itemsize*size);
            DEVICE_TO_HOST(_pdata, (int *)_pdata_device, itemsize*size);
            _display(_pdata, ptr_buf->ndim, ptr_buf->shape);
            free(_pdata);
            break;
            }
        case 'f':{
            ssize_t itemsize = ptr_buf->itemsize;
            ssize_t size = ptr_buf->size;
            float *_pdata = (float *)malloc(itemsize*size);
            DEVICE_TO_HOST(_pdata, (float *)_pdata_device, itemsize*size);
            _display(_pdata, ptr_buf->ndim, ptr_buf->shape);
            free(_pdata);
            break;
            }
        case 'd':{
            ssize_t itemsize = ptr_buf->itemsize;
            ssize_t size = ptr_buf->size;
            double *_pdata = (double *)malloc(itemsize*size);
            DEVICE_TO_HOST(_pdata, (double *)_pdata_device, itemsize*size);
            _display(_pdata, ptr_buf->ndim, ptr_buf->shape);
            free(_pdata);
            break;
            }
        case NULL:
           printf("current Array obj has no data concent, cannot execute display_cuda()!\n");
           break;
        default:
            throw std::runtime_error("current version only support int32;float32;float64!");
            break;
    }
}
