/*
the file imitates sklearn.preprocessing.data
*/
#include<stdio.h>
#include<cuda_runtime.h>
#include<limits>
#include<assert.h>
#include<cmath>

#include "ML/preprocessing/data.h"

#include "common/helper.h"

#define MAX(x,y) ((x)>(y) ? (x): (y))
#define MIN(x,y) ((x)>(y) ? (y): (x))


/* get the [n by m] matrix's maxValue vector and minValue vector by col dimension,
means maxVal vector is [1 by m], minVal vector is [1 by m]*/
template<class T> __global__ void
get_minmax(T *mat, T *min, T *max, unsigned int col, unsigned int row, T min_val, T max_val){
    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int thread_idx = (gridDim.x * blockDim.x)*idy +idx;

    // launch col numbers cuda threads to handle the matrix.
    // each cuda thread handle one col dimension
    if (thread_idx < col){
        T min_l = max_val;
        T max_l = min_val;
    
        T tmp = (T)0;
        for (unsigned int i = 0; i < row; i++){
            tmp = *(mat + col*i + thread_idx);
            min_l = MIN(tmp, min_l);
            max_l = MAX(tmp, max_l);
        } 
        min[thread_idx] = min_l;
        max[thread_idx] = max_l;
    }
}

/*normaliza the [n by m] matrix, use col cuda threads*/
template<class T> __global__ void
minmax_scale_cuda(T *mat, T *min, T *max, unsigned int col, unsigned int row, T feature_min, T feature_max){
/* feature_range should bigger than 0*/
    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int thread_idx = (gridDim.x * blockDim.x)*idy +idx;

    if (thread_idx < col){
       T min_l = min[thread_idx];
       T max_l = max[thread_idx];

       T range = max_l - min_l;
       
       T feature_range_l = feature_max - feature_min; // should not be 0, if it's 0, then replace with 1
       assert(feature_range_l > 0 );
       float scale = feature_range_l / range; 
       float min_  = feature_min - min_l * scale;
       
       T tmp = (T)0;
       for(unsigned int i = 0; i<row; i++){
           tmp = *(mat + col*i + thread_idx);
           tmp *= scale;
           tmp += min_;
           *(mat + col*i + thread_idx) = (T)tmp;

       }

    }

}

template<class T> int
minmax_scale_cpu(T *mat, unsigned int col, unsigned int row, T feature_min, T feature_max){

    T *mat_d = NULL;
    size_t size_mat = sizeof(T)*col*row;
    CHECK_CALL(cudaMalloc((void **)&mat_d, size_mat));
    CHECK_CALL(cudaMemcpy(mat_d, mat, size_mat, cudaMemcpyHostToDevice));
 
    T *min_d = NULL;
    size_t size_min = sizeof(T)*col;
    CHECK_CALL(cudaMalloc((void **)&min_d, size_min));
//    cudaMemcpy(min_d, min, size_min, cudaMemcpyHostToDevice);

    T *max_d = NULL;
    size_t size_max = sizeof(T)*col;
    CHECK_CALL(cudaMalloc((void **)&max_d, size_max));
//    cudaMemcpy(max_d, max, size_max, cudaMemcpyHostToDevice);


    // max blockdim is 65536,so max col not bigger than 65536*65536*256.
    unsigned int threaddim = 32;
    int blockdim = MAX(ceil(sqrt( ceil(col/threaddim) )),1);
    dim3 grid_size(blockdim, blockdim);
    dim3 block_size(1,threaddim); 

    T min_val = std::numeric_limits<T>::min();
    T max_val = std::numeric_limits<T>::max();

    get_minmax<T><<<grid_size, block_size>>>(mat_d, min_d, max_d, col, row, min_val, max_val);
    cudaDeviceSynchronize();

    minmax_scale_cuda<T><<<grid_size, block_size>>>(mat_d, min_d, max_d, 
                                     col, row, feature_min, feature_max);

    CHECK_CALL(cudaMemcpy(mat, mat_d, size_mat, cudaMemcpyDeviceToHost));

    cudaFree(mat_d); 
    cudaFree(min_d);
    cudaFree(max_d); 
    return 0;
}

//
//int
//main(){
//
//    float mat[4][2] = {{-1,2},{-0.5,6},{0,10},{1,18}};
//
//    float *p = &mat[0][0];
//    minmax_scale_cpu<float>(p, 2, 4, 3, 6);
//
//    for (int i=0; i<4; i++){
//        for (int j=0; j<2; j++){
//           printf("%d %d vale %f\n",i,j,mat[i][j]);
//        }
//    }
//}
