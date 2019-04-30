#include "common/buffer_info_ex.h"
#include "common/malloc_free.h"

Buf& buffer_info_ex::cuda(){

    host_to_device(*this);
    return *this;
}

Array & Array::cuda(){
    this->cuda();
    return *this;

}

