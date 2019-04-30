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

class Array{

    Buf *ptr_buf;

    Array() {}

    Array(py::array_t<float> &array){
        auto array_info = array.requests();

        if (array_info.format != py::format_descriptor<float>::format())
            throw std::runtime_error("Incompatible format: excepted a float32 array!");
        if (array_info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        this->ptr_buf = new Buf(array_info);
    }

    Array & cuda();

};
