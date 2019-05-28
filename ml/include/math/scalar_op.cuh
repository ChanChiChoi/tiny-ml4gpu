#pragma once

//#include <string.h>
#include <math_functions.hpp>
#include "ml/include/math/scalar_op_def.h"

# ifndef __SCALAR_OP__
# define __SCALAR_OP__


//=====================
/*
template === one scalar operation
*/
template<typename T> __device__ T
scalar_sqrt(T x){
    return sqrt(x);
}

template<typename T> __device__ T
scalar_invsqrt(T x){
    return 1/sqrt(x);
}

template<typename T> __device__ T
scalar_operation1(T x, const int op){
  /*
   this function used to be entrance of how to handle one scalar
   1 - sqrt(x)

  */
  T ans = T(0);
  if (op == SCALAR_ONE_SQRT){
      ans = scalar_sqrt<T>(x);
  }else if(op == SCALAR_ONE_INVSQRT){
      ans = scalar_invsqrt<T>(x);
  }
  return ans;
}




/*
template === two scalar operation
*/
template<typename T> __device__ T
scalar_multiply(T x, T y){
  return x*y;
}

template<typename T> __device__ T
scalar_mse(T x, T y){
    T tmp = abs(x-y);
    return tmp*tmp;
}

template<typename T> __device__ T
scalar_divide(T x, T y){
  return x-y;
}


template<typename T> __device__ T
scalar_add(T x, T y){
  return x+y;
}

template<typename T> __device__ T
scalar_gaussian(T x, T sigma){
  // T should not be int data type, in case of return 0
  return exp(-x*x/(2*sigma*sigma));
   
}


template<typename T> __device__ T
scalar_operation2(T x, T y, const int op){
  /*
   this function used to be entrance of how to handle two scalar
   1 - x * y
   2 - |x-y|^2

  */
  T ans = T(0);
  if (op == SCALAR_TWO_MUL){
      ans = scalar_multiply<T>(x,y);
  }else if(op == SCALAR_TWO_MSE){
      ans = scalar_mse<T>(x,y); 
  }else if(op == SCALAR_TWO_DIVIDE){
      ans = scalar_divide<T>(x,y);
  }else if(op == SCALAR_TWO_GAUSSIAN){
      ans = scalar_gaussian<T>(x,y);
  }else if(op == SCALAR_TWO_ADD){
      ans = scalar_add<T>(x,y);
  }
  return ans;
}
# endif
