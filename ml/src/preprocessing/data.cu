#include <stdio.h>
#include <cuda_runtime.h>
#include <math_functions.hpp>
#include "common/include/type.h"
#include "common/include/common.h"

//====================template
// calc the mean by row dimension
template<typename T> __global__ void
mean_by_rows(T *mat_device, T *mean_vec, u32 rows, u32 cols){

    u32 idy = blockIdx.y*blockDim.y + threadIdx.y;
    u32 idx = blockIdx.x*blockDim.x + threadIdx.x;

    u32 thread_idx = idy*(gridDim.x*blockIdx.x) + idx;

    if(thread_idx >= cols)
        return ;

    T mean = (T)0;
    T cur_val = (T)0;
    for (u32 i = 0; i < rows; i++){
        cur_val = mat_device[i*cols+thread_idx];
        // in case of sum is too big
        mean = mean*((float)i/(i+1)) +  cur_val/(double)(i+1);
    }
    mean_vec[thread_idx] = mean;
    
}

// each row subtract the mean vector
template<typename T> __global__ void
zero_mean_by_rows(T *mat_device, T *mean_vec, u32 rows, u32 cols){

    u32 idy = blockIdx.y*blockDim.y + threadIdx.y;
    u32 idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < cols && idy < rows){
        u32 val_idx = idy*cols + idx;
        mat_device[val_idx] -= mean_vec[idx];
    }

}

// calc the std by rows dimension
template<typename T> __global__ void
std_by_rows(T *mat_device, T *mean_vec, T *std_vec, u32 rows, u32 cols){

    u32 idy = blockIdx.y*blockDim.y + threadIdx.y;
    u32 idx = blockIdx.x*blockDim.x + threadIdx.x;

    u32 thread_idx = idy*(gridDim.x*blockIdx.x) + idx;

    if (thread_idx >= cols)
        return ;
    
    T cur_val;
    T std = T(0);
    for (size_t i = 0; i< rows; i++){
        cur_val = mat_device[i*cols+thread_idx];
        // in case of std sum is bigger than limits
        std = std*((float)i/(i+1)) +  cur_val*cur_val/(double)(i+1);
    }

    std_vec[thread_idx] = sqrt(std);

} 


// each row divide the std vector
template<typename T> __global__ void
one_std_by_rows(T *mat_device, T *std_vec, u32 rows, u32 cols){

    u32 idy = blockIdx.y*blockDim.y + threadIdx.y;
    u32 idx = blockIdx.x*blockDim.x + threadIdx.x;


    if(idx < cols && idy < rows){
        u32 val_idx = idy*cols + idx;
        mat_device[val_idx] /= std_vec[idx];
    }

}

//===========launch
//export the function for be called by host
void
mean_by_rows_launch(float *mat_device, float *mean_device, u32 rows, u32 cols){


    const u32 COLS = 256;
    dim3 grid0( MAX(1,ceil(double(cols)/COLS)) );
    dim3 block0(COLS);

    mean_by_rows<float><<<grid0, block0>>>(mat_device, mean_device, rows, cols);

    const u32 block_size = 32;
    dim3 block1(block_size, block_size);

    dim3 grid1(MAX(1, ceil((double)cols/block_size)),
              MAX(1, ceil((double)rows/block_size)));

    zero_mean_by_rows<float><<<grid1,block1>>>(mat_device, mean_device, rows, cols);
}

void
normalization_by_rows_launch(float *mat_device, float *mean_device, float *std_device, u32 rows, u32 cols){

    mean_by_rows_launch(mat_device, mean_device, rows, cols);
    
    const u32 COLS = 256;
    dim3 grid0( MAX(1, ceil(double(cols)/COLS)) );
    dim3 block0(COLS);

    std_by_rows<float><<<grid0, block0>>>(mat_device, mean_device, std_device, rows, cols);

    const u32 block_size = 32;
    dim3 block1(block_size, block_size);
    dim3 grid1(MAX(1, ceil(double(cols)/block_size)),
               MAX(1, ceil(double(rows)/block_size)));

    one_std_by_rows<float><<<grid1, block1>>>(mat_device, std_device, rows, cols);
    

}
//===============export to host
void
mean_by_rows_cpu(float *mat_device, float *mean_device, u32 rows, u32 cols){

    mean_by_rows_launch(mat_device, mean_device, rows, cols);
}

void
normalization_by_rows_cpu(float *mat_device, float *mean_device, float *std_device, u32 rows, u32 cols){
    normalization_by_rows_launch(mat_device, mean_device, std_device, rows, cols);
}

