#include <assert.h>
#include <stdio.h>
#include <math_functions.hpp>
#include "common/include/type.h"
#include "common/include/common.h"
//#include "common/include/malloc_free.h"
#include "ml/include/math/matrix.h"

# define TILE_HEIGHT 32
# define TILE_WIDTH 32

//template ===
//matrix_mul

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
scalar_operation(T x, T y, const int op){
  /*
   this function used to be entrance of how to handle two scalar
   1 - x * y
   2 - |x-y|^2

  */
  T ans = T(0);
  if (op == 1){
      ans = scalar_multiply<T>(x,y);
  }else if(op == 2){
      ans = scalar_mse(x,y); 
  }
  return ans;
}


template<typename T> __global__ void
matrix_mul(T * Md, u32 Row_Md, u32 Col_Md,
           T * Nd, u32 Row_Nd, u32 Col_Nd,
           T * Pd, u32 Row_Pd, u32 Col_Pd,
           const int op = 1
           ){
    
    /*
     each thread has two task:
    1 - fetch data into shared mem;
    2 - calc the data by steps , then get result of Pd

*/
    // row = height = x;   
    // col = width = y
    assert(Col_Md == Row_Nd);

    __shared__ T Mds[TILE_HEIGHT][TILE_WIDTH];
    __shared__ T Nds[TILE_HEIGHT][TILE_WIDTH];

    u32 bx = blockIdx.x;
    u32 by = blockIdx.y;
    u32 tx = threadIdx.x;
    u32 ty = threadIdx.y;

    // split Md by TILE_WIDTH, so each blocksize equal TILE_WIDTH
    // current we create the Row,Col in Pd
    u32 Row = by*TILE_HEIGHT + ty;
    u32 Col = bx*TILE_WIDTH + tx;

    // here should not use  "(Row < Row_Pd && Col < Col_Pd)", because we need other thread
    // to fetch data into shared mem.

    T Pvalue = 0;

    
    // for cur tx,ty only care Col of Md and Row of Nd
    for(u32 m = 0; m < ceil((double)Col_Md/TILE_WIDTH); ++m){
        // get the data again and again
        // if cur tx,ty is exceend of Md,Nd, then it should be exit early,
        // so it will not run here
        const u32 ind_bef_Md = Row*Col_Md;
        const u32 ind_x_Md = m*TILE_WIDTH + tx;

        const u32 ind_y_Nd = m*TILE_HEIGHT + ty;

        // if cur x is exceed col of md, then skip
        if (ind_x_Md < Col_Md)
            Mds[ty][tx] = Md[ind_bef_Md + ind_x_Md];
 
        // if cur y is exceed row of nd, then skip
        if (ind_y_Nd  < Row_Nd)
            Nds[ty][tx] = Nd[ind_y_Nd*Col_Nd + Col];

        // if cur thread can do nothing, then exit
        if (ind_x_Md >= Col_Md && ind_y_Nd >= Row_Nd)
            return ;

        __syncthreads();

       // if cur thread' task contain create pd result, it need handle follow code
       if (Row < Row_Pd && Col < Col_Pd){ 

           u32 ind_max_TILE;
           if ((m+1)*TILE_WIDTH <= Col_Md)
               ind_max_TILE = TILE_WIDTH;
           else
               ind_max_TILE = Col_Md - m*TILE_WIDTH;
           // calc the point
           for(u32 k = 0; k < ind_max_TILE; ++k){
              //Pvalue += Mds[ty][k] * Nds[k][tx];
              Pvalue += scalar_operation(Mds[ty][k], Nds[k][tx],op);
           }
       }

       __syncthreads();

   } 

   if(Row >= Row_Pd || Col >= Col_Pd)
        return ;
   // put the result into origin location of Pd
   Pd[Row*Col_Pd + Col] = Pvalue;
}

template<typename T> void
matrix_mul_launch(T * Md, u32 Row_Md, u32 Col_Md,
           T * Nd, u32 Row_Nd, u32 Col_Nd,
           T * Pd, u32 Row_Pd, u32 Col_Pd,
           const int op = 1){

    dim3 grid(MAX(1, (size_t)ceil((double)Col_Pd/TILE_HEIGHT)),
              MAX(1, (size_t)ceil((double)Row_Pd/TILE_WIDTH)) );
    dim3 block(TILE_WIDTH, TILE_HEIGHT);

    matrix_mul<T><<<grid, block>>>(Md, Row_Md, Col_Md,
           Nd, Row_Nd, Col_Nd,
           Pd, Row_Pd, Col_Pd,
           op);
}


template<typename T> __global__ void
matrix_transpose(T * mat_src, u32 Row_src, u32 Col_src,
                 T * mat_dst, u32 Row_dst, u32 Col_dst){

    assert(Row_src*Col_src == Row_dst*Col_dst);
    u32 idy = blockIdx.y*blockDim.y + threadIdx.y;
    u32 idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idy >= Row_src || idx >= Col_src)
        return ;

    mat_dst[idx*Col_dst + idy] = mat_src[idy*Col_src + idx];

}


template<typename T> void
matrix_transpose_launch(T *mat_src, u32 Row_src, u32 Col_src,
                 T * mat_dst, u32 Row_dst, u32 Col_dst){

    dim3 grid(MAX(1, (size_t)ceil((double)Col_src/TILE_HEIGHT)),
              MAX(1, (size_t)ceil((double)Row_src/TILE_WIDTH)) );
    dim3 block(TILE_WIDTH, TILE_HEIGHT);

    matrix_transpose<T><<<grid, block>>>(mat_src, Row_src, Col_src,
                                   mat_dst, Row_dst, Col_dst);
}

template<typename T>__global__ void
matrix_divide_scalar(T *mat, u32 Row, u32 Col, u32 scalar){

    u32 idy = blockIdx.y*blockDim.y + threadIdx.y;
    u32 idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idy >= Row || idx >= Col)
        return ;

    mat[idy*Col+idx] /= scalar;
}

template<typename T> void
matrix_divide_scalar_launch(T *mat, u32 Row, u32 Col, u32 scalar){

    dim3 grid(MAX(1, (size_t)ceil((double)Col/TILE_HEIGHT)),
              MAX(1, (size_t)ceil((double)Row/TILE_WIDTH)) );
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
  
    matrix_divide_scalar<T><<<grid, block>>>(mat, Row, Col, scalar);

}

template<typename T> __global__ void
matrix_subblock(T *big, u32 Row_big, u32 Col_big,
                T *small, u32 Row_sm, u32 Col_sm,
                u32 rmin, u32 cmin, u32 rmax, u32 cmax){
    // rmin base on 0.
    u32 idy = blockIdx.y*blockDim.y + threadIdx.y;
    u32 idx = blockIdx.x*blockDim.x + threadIdx.x;
 
    assert(rmax - rmin == Row_sm);
    assert(cmax - cmin == Col_sm);

    if(idy >= Row_sm || idx >= Col_sm)
        return ;

    small[idy*Col_sm+idx] = big[(rmin+idy)*Col_big+cmin+idx];
}

template<typename T> void
matrix_subblock_launch(T *big, u32 Row_big, u32 Col_big,
                       T *small, u32 Row_sm, u32 Col_sm,
                       u32 rmin, u32 cmin, u32 rmax, u32 cmax){

    dim3 grid(MAX(1, (size_t)ceil((double)Col_sm/TILE_HEIGHT)),
              MAX(1, (size_t)ceil((double)Row_sm/TILE_WIDTH)) );
    dim3 block(TILE_WIDTH, TILE_HEIGHT);

    matrix_subblock<T><<<grid, block>>>(big, Row_big, Col_big,
                                       small, Row_sm, Col_sm,
                                      rmin, cmin, rmax, cmax);

}
// ==========cov


//================ 
void
matrix_mul_cpu(float *Md, u32 Row_Md, u32 Col_Md,
               float *Nd, u32 Row_Nd, u32 Col_Nd,
               float *Pd, u32 Row_Pd, u32 Col_Pd,
               const int op = 1){

    matrix_mul_launch<float>(Md, Row_Md, Col_Md,
               Nd, Row_Nd, Col_Nd,
               Pd, Row_Pd, Col_Pd,
               op);

}

void
matrix_transpose_cpu(float *mat_src, u32 Row_src, u32 Col_src,
                     float * mat_dst, u32 Row_dst, u32 Col_dst){

    matrix_transpose_launch<float>(mat_src, Row_src, Col_src,
                      mat_dst, Row_dst, Col_dst);
}

void
matrix_divide_scalar_cpu(float *mat, u32 Row, u32 Col, u32 scalar){

    matrix_divide_scalar_launch<float>(mat, Row, Col, scalar);
}

void
matrix_subblock_cpu(float *big, u32 Row_big, u32 Col_big,
                float *small, u32 Row_sm, u32 Col_sm,
                u32 rmin, u32 cmin, u32 rmax, u32 cmax){

    matrix_subblock_launch<float>(big, Row_big, Col_big,
                           small, Row_sm, Col_sm,
                           rmin, cmin, rmax, cmax);
}

//void
//cov_cpu(float *mat, u32 Row_mat, u32 Col_mat,
//        float *mat_cov, u32 Row_mat_cov, u32 Col_mat_cov){
//    
//    //1 - malloc one matrix
//    size_t size = sizeof(float)*Row_mat*Col_mat;
//    float *mat_T_device = NULL;
//    mat_T_device = DEVICE_MALLOC(mat_T_device, size);
//
//    //2 - transpose
//    u32 Row_mat_T = Col_mat;
//    u32 Col_mat_T = Row_mat;
//    matrix_transpose_cpu(mat,Row_mat, Col_mat,
//                  mat_T_device, Row_mat_T, Col_mat_T);
//
//    //3 - matrix_mul
//
//    matrix_mul_cpu(mat_T_device,Row_mat_T, Col_mat_T,
//                   mat, Row_mat, Col_mat,
//                   mat_cov, Row_mat_cov, Col_mat_cov,1);
//
//    DEVICE_FREE(mat_T_device);
//
//    //4 - divide (n-1) samples;
//    size_t n_1 = MAX(1,Row_mat-1);
//    matrix_divide_scalar_cpu(mat_cov, Row_mat_cov, Col_mat_cov, n_1);
//
//}
