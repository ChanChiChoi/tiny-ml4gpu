#include <stdio.h>
#include <cuda_runtime.h>
#include <math_functions.hpp>
#include "common/include/type.h"
#include "common/include/common.h"

//====================template
// calc the mean by row dimension
template<typename T> __global__ void
mean_by_rows(T *mat_device, T *mean_vec, u32 rows, u32 cols){

    u32 idy = blockIdx.y*gridDim.y + threadIdx.y;
    u32 idx = blockIdx.x*gridDim.x + threadIdx.x;

    u32 thread_idx = idy*(gridDim.x*blockIdx.x) + idx;

    if(thread_idx < cols){
        T mean = (T)0;
        T cur_val = (T)0;
        for (u32 i = 0; i < rows; i++){
            cur_val = mat_device[i*cols+thread_idx];
            // in case of sum is too big
            mean = mean*((float)i/(i+1)) +  cur_val/(double)(i+1);
        }

        mean_vec[thread_idx] = mean;
    }
}

// each row subtract the mean vector
template<typename T> __global__ void
zero_mean_by_rows(T *mat_device, T *mean_vec, u32 rows, u32 cols){

    u32 idy = blockIdx.y*gridDim.y + threadIdx.y;
    u32 idx = blockIdx.x*gridDim.x + threadIdx.x;


    if(idx < cols && idy < rows){
        u32 val_idx = idy*cols + idx;
        mat_device[val_idx] -= mean_vec[idx];
    }

}

// calc the std by rows dimension
template<typename T> __global__ void
std_by_rows(T *mat_device, T *mean_vec, T *std_vec, u32 rows, u32 cols){

    u32 idy = blockIdx.y*gridDim.y + threadIdx.y;
    u32 idx = blockIdx.x*gridDim.x + threadIdx.x;

    u32 thread_idx = idy*(gridDim.x*blockIdx.x) + idx;
    if (thread_idx >= cols)
        return ;
    
    T cur_mean = mean_vec[thread_idx];
    T cur_val;
    T dif;
    T std;
    for (size_t i = 0; i< rows; i++){
        cur_val = mat_device[i*cols+thread_idx];
        dif = abs(cur_val - cur_mean);
        // in case of std sum is bigger than limits
        std = std*((float)i/(i+1)) +  dif*dif/(double)(i+1);
    }
    std_vec[thread_idx] = sqrt(std);

} 


// each row divide the std vector
template<typename T> __global__ void
one_std_by_rows(T *mat_device, T *std_vec, u32 rows, u32 cols){

    u32 idy = blockIdx.y*gridDim.y + threadIdx.y;
    u32 idx = blockIdx.x*gridDim.x + threadIdx.x;


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
    dim3 grid0( MAX(1,ceil((double)cols/COLS)) );
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
    dim3 grid0( MAX(1, ceil(double)cols/COLS));
    dim3 block0(COLS);

    std_by_rows<float><<<grid0, block0>>>(mat_device, mean_vec, std_device, rows, cols);

    const u32 block_size = 32;
    dim3 block1(block_size, block_size);
    dim3 grid1(MAX(1, ceil(double(cols)/block_size)),
               MAX(1, ceil(double(rows)/blocks_size)));

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
/*
int
main(){

    size_t size = 200*sizeof(float);
    float *mat = (float *)malloc(size);
    for(u32 i=0;i<200;i++)
        mat[i] = i;


    float *mat_device = host_to_device_malloc(mat,size);

    size_t size1 = 50*sizeof(float);
    float *mean = (float *)malloc(size1);
    float *mean_device = host_to_device_malloc(mean,size1);

    auto t0 = high_resolution_clock::now();
    mean_by_rows_cpu(mat_device,mean_device, 4,50);
    cudaDeviceSynchronize();
    auto t1 = high_resolution_clock::now();
    device_to_host_free(mat,mat_device,size);
    device_to_host_free(mean,mean_device,size1);
    printf("take time %d\n",duration_cast<milliseconds>(t1-t0).count());

//
    for(u32 i=0; i<20;i++){
        printf("val %d %f\n",i,mat[i]);
    }

}
*/
