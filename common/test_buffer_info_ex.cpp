/*
g++ test_buffer_info_ex.cpp -I../pybind11/include -I../ `cd ../pybind11 &&  python3 -m pybind11 --includes` -std=c++11
*/
#include "buffer_info_ex.h"
#include <string>

int
main(){

  Array *ptr = new Array{
   size_t(10),
   size_t(20),
   std::string(1,'f')
   };



  return 0;
}