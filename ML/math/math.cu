#include <assert.h>
#include <stdio.h>
#include <math_functions.hpp>
#include "common/type.h"
#include "common/common.h"
#include "common/malloc_free.h"
#include <chrono>

# define TILE_HEIGHT 32
# define TILE_WIDTH 32

//matrix_multiplication ===
template<typename T> __global__ void
matrix_mul(T * Md, u32 Row_Md, u32 Col_Md,
           T * Nd, u32 Row_Nd, u32 Col_Nd,
           T * Pd, u32 Row_Pd, u32 Col_Pd){
    
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
    
    //printf("[bx by tx ty row col][%d %d %d %d %d %d]\n",bx,by,tx,ty,Row,Col);
    // if current thread is exceed the Pd matrix, then return void
    if(Row >= Row_Pd ||Col >= Col_Pd)
        return ;

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

        __syncthreads();

       u32 ind_max_TILE;
       if ((m+1)*TILE_WIDTH <= Col_Md)
           ind_max_TILE = TILE_WIDTH;
       else
           ind_max_TILE = Col_Md - m*TILE_WIDTH;
       // calc the point
       for(u32 k = 0; k < ind_max_TILE; ++k)
          Pvalue += Mds[ty][k] * Nds[k][tx];

       __syncthreads();

   } 

   // put the result into origin location of Pd
   Pd[Row*Col_Pd + Col] = Pvalue;
}

template<typename T> void
matrix_mul_launch(T * Md, u32 Row_Md, u32 Col_Md,
           T * Nd, u32 Row_Nd, u32 Col_Nd,
           T * Pd, u32 Row_Pd, u32 Col_Pd){

    dim3 grid(MAX(1, (size_t)ceil((double)Col_Pd/TILE_HEIGHT)),
              MAX(1, (size_t)ceil((double)Row_Pd/TILE_WIDTH)) );
    dim3 block(TILE_WIDTH, TILE_HEIGHT);

    matrix_mul<T><<<grid, block>>>(Md, Row_Md, Col_Md,
           Nd, Row_Nd, Col_Nd,
           Pd, Row_Pd, Col_Pd);
}


template<typename T> __global__ void
matrix_transpose(T * mat_src, u32 Row_src, u32 Col_src,
                 T * mat_dst, u32 Row_dst, u32 Col_dst){

    assert(Row_src*Col_src == Row_dst*Col_dst);
    u32 idy = blockIdx.y*gridDim.y + threadIdx.y;
    u32 idx = blockIdx.x*gridDim.x + threadIdx.x;

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

    u32 idy = blockIdx.y*gridDim.y + threadIdx.y;
    u32 idx = blockIdx.x*gridDim.x + threadIdx.x;

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
// ==========cov


//================ 
void
matrix_mul_cpu(float *Md, u32 Row_Md, u32 Col_Md,
               float *Nd, u32 Row_Nd, u32 Col_Nd,
               float *Pd, u32 Row_Pd, u32 Col_Pd){

    matrix_mul_launch<float>(Md, Row_Md, Col_Md,
               Nd, Row_Nd, Col_Nd,
               Pd, Row_Pd, Col_Pd);
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
cov_cpu(float *mat, u32 Row_mat, u32 Col_mat,
        float *mat_cov, u32 Row_mat_cov, u32 Col_mat_cov){
    
    //1 - malloc one matrix
    size_t size = sizeof(float)*Row_mat*Col_mat;
    float *mat_T_device = device_malloc<float>(size);

    //2 - transpose
    u32 Row_mat_T = Col_mat;
    u32 Col_mat_T = Row_mat;
    matrix_transpose_cpu(mat,Row_mat, Col_mat,
                  mat_T_device, Row_mat_T, Col_mat_T);

    //3 - matrix_mul

    matrix_mul_cpu(mat_T_device,Row_mat_T, Col_mat_T,
                   mat, Row_mat, Col_mat,
                   mat_cov, Row_mat_cov, Col_mat_cov);

    device_free<float>(mat_T_device);

    //4 - divide (n-1) samples;
    size_t n_1 = MAX(1,Row_mat-1);
    matrix_divide_scalar_cpu(mat_cov, Row_mat_cov, Col_mat_cov, n_1);

}
//int
//main(){
//
//   size_t rowm = 1024,colm = 1024;
//   size_t rown = colm, coln = 1024;
//   size_t rowp = rowm, colp = coln;
//   
//   size_t size = sizeof(float)*rowm*colm;
//   float *md = (float *)malloc(size);
//   for(int i=0;i<rowm*colm;i++)
//     md[i] = i%1000;
//   float *md_d = host_to_device_malloc(md, size);
//
//   size_t size1 = sizeof(float)*rown*coln;
//   float *nd = (float *)malloc(size1);
//   for(int i=0;i<rown*coln;i++)
//      nd[i]=2;
//   float *nd_d = host_to_device_malloc(nd, size1);
//   
//   size_t size2 = sizeof(float)*rowp*colp;
//   float *pd = (float *)malloc(size2);
//   float *pd_d = host_to_device_malloc(pd, size2);
//
//   auto t1 = high_resolution_clock::now();
//   matrix_mul_cpu(md_d,rowm,colm,
//                       nd_d,rown,coln,
//                       pd_d,rowp,colp);
//  cudaDeviceSynchronize(); 
//  auto t2 = high_resolution_clock::now();
//  printf("take time %d\n",duration_cast<milliseconds>(t2-t1).count());
//
//   device_to_host_free(pd,pd_d,size2);
////   for(int i = 0; i<rowp; i++){
////      printf("[%d] ",i+1);
////      for(int j=0;j<colp;j++)
////          printf("%d ",(int)pd[i*colp+j]);
////      printf("\n");
////   }
//  return 0;
//}
