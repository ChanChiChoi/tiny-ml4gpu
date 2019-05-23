#include <math_functions.hpp>
#include <assert.h>
#include "common/include/type.h"
#include "common/include/common.h"
#include "ml/include/math/vector.h"

# define TILE_HEIGHT 32
# define TILE_WIDTH 32

# define BLOCK_LENGTH 256


/*
template: vector_repeat_by_rows
*/
template<typename T> __global__ void
vector_repeat_by_rows(T *mat_device, u32 rows_mat, u32 cols_mat, 
              T *vector, u32 cols_vec){

    // TODO: this function can be optimization
    assert(cols_mat == cols_vec);

    u32 idy = blockIdx.y*blockDim.y + threadIdx.y;
    u32 idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= cols_mat || idy >= rows_mat)
        return ;

    mat_device[idy*cols_mat+idx] = vector[idx];

}

template<typename T> void
vector_repeat_by_rows_launch(T *mat_device, u32 rows_mat, u32 cols_mat,
              T *vector_device, u32 cols_vec){
    
    dim3 grid(MAX(1, (size_t)ceil((double)cols_mat/TILE_HEIGHT)),
              MAX(1, (size_t)ceil((double)rows_mat/TILE_WIDTH)) );
    dim3 block(TILE_WIDTH, TILE_HEIGHT);

    vector_repeat_by_rows<T><<<grid, block>>>(mat_device, rows_mat, cols_mat,
                        vector_device, cols_vec);
}


/*
template: vector_sum
*/
template<typename T> __global__ void
vector_sum(T *vec, u32 len, T *res){
  //TODO: ???????????????
  __shared__ T block[LENGTH];
  

}

template<typename T> void
vector_sum_launch(T *vec, u32 len, T *res){

    dim3 grid();
    dim3 block(LENGTH);
  
    vector_sum<T><<<grid, block>>>(vec, len, res);
}

/*
template: vector_op_self
*/
template<typename T> __global__ void
vector_op_self(T *vec, u32 len, const char *op){

    u32 idx = blockIdx.x*blockDim.x+threadIdx.x;
  
    if (idx >= len)
        return ;

    T tmp = vec[idx];
    vec[idx] = scalar_operation1(tmp, op);
} 

template<typename T> __global__ void
vector_op_self_launch(T *vec, u32 len, const char *op){

    dim3 grid(MAX(1, ceil(double(len)/BLOCK_LENGTH)));
    dim3 block(BLOCK_LENGTH);

    vector_op_self<T><<<grid, block>>>(vec, len, op);
}

//=================cpu

void
vector_repeat_by_rows_cpu(float *mat_device, u32 rows_mat, u32 cols_mat,
                          float *vector_device,u32 cols_vec){
    vector_repeat_by_rows_launch(mat_device, rows_mat, cols_mat,
                              vector_device, cols_vec);
}


void
vector_sum_cpu(float *vec, u32 len, float *res){
    vector_sum_launch<float>(vec, len, res);
}

void
vector_invsqrt_self_cpu(float *vec, u32 len){
    vector_op_self_launch<float>(vec, len, "invsqrt");
}
