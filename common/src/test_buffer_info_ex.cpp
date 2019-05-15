/*
g++ test_buffer_info_ex.cpp -I../../pybind11/include -I../../ `cd ../../pybind11 &&  python3 -m pybind11 --includes` -std=c++11
*/
#include <string>
#include <stdio.h>
#include "common/include/buffer_info_ex.h"

int
main(){

  Array *ptr = new Array{
   size_t(10),
   size_t(20),
   std::string(1,'f')
   };

   printf("shape rows:%d\n",ptr->ptr_buf->shape[0]);

  Array *ptr1 = new Array{
       nullptr,nullptr,nullptr,
       2,{3,4},std::string(1,'f'),32,12,{32,32}
   };

   printf("shape rows:%d\n",ptr1->ptr_buf->shape[0]);
//   ptr1->cpu();
  

  return 0;
}
