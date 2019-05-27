#pragma once

//#include <string.h>
#include <math_functions.hpp>
#include "common/include/type.h"
#include "common/include/common.h"

# ifndef __SCALAR_OP__
# define __SCALAR_OP__
NAMESPACE_BEGIN(m4g)
__device__ int 
strcmp(const char *x, const char *y){
  
    assert(x != nullptr && y != nullptr);
    while( *y != '\0'){
        if (*x < *y){
            return -1;
        }else if (*x > *y){
            return 1;
        }
        x++;
        y++;
    }
    return 0;
}

NAMESPACE_END(m4g)

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
scalar_operation1(T x, const char *op){
  /*
   this function used to be entrance of how to handle one scalar
   1 - sqrt(x)

  */
  T ans = T(0);
  if (m4g::strcmp(op,"sqrt") == 0){
      ans = scalar_sqrt<T>(x);
  }else if(m4g::strcmp(op, "invsqrt") == 0){
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
scalar_operation2(T x, T y, const char  *op){
  /*
   this function used to be entrance of how to handle two scalar
   1 - x * y
   2 - |x-y|^2

  */
  T ans = T(0);
  if (m4g::strcmp(op,"mul") == 0){
      ans = scalar_multiply<T>(x,y);
  }else if(m4g::strcmp(op, "mse") == 0){
      ans = scalar_mse<T>(x,y); 
  }else if(m4g::strcmp(op,"divide") == 0){
      ans = scalar_divide<T>(x,y);
  }else if(m4g::strcmp(op, "gaussian") == 0){
      ans = scalar_gaussian<T>(x,y);
  }else if(m4g::strcmp(op, "add") == 0){
      ans = scalar_add<T>(x,y);
  }
  return ans;
}
# endif
