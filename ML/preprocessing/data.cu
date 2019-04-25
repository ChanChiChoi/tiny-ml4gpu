/*
the file imitates sklearn.preprocessing.data
*/
#include<cuda_runtime.h>
#include<limit>
#include<assert.h>
#include<cmath>

#define MAX(x,y) ((x)>(y) ? (x): (y))
#define MIN(x,y) ((x)>(y) ? (y): (x))


/* get the [n by m] matrix's maxValue vector and minValue vector by col dimension,
means maxVal vector is [1 by m], minVal vector is [1 by m]*/
template<class T> __global__ void
get_minmax(T *mat, T *min, T *max, unsigned int col, unsigned int row){
    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int thread_idx = (gridDim.x * blockDim.x)*idy +idx;

    // launch col numbers cuda threads to handle the matrix.
    // each cuda thread handle one col dimension
    if (thread_idx < col){
        T min_l = std::numeric_limits<T>::max();
        T max_l = std::numeric_limits<T>::min();
    
        T tmp = (T)0;
        for (unsigned int i = 0; i < row; i++){
            tmp = *(mat + row*i + thread_idx)
            min_l = MIN(tmp, min_l)
            max_l = MAX(tmp, max_l)
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
       T min_l = min[thread_idx]
       T max_l = max[thread_idx]

       T range = max_l - min_l;
       
       T feature_range_l = feature_max - feature_min; // should not be 0, if it's 0, then replace with 1
       assert(feature_range_l > 0 );
       float scale = feature_range_l / range; 
       float min_  = feature_min - min_l * scale;
       
       T tmp = (T)0;
       for(unsigned int i = 0; i<row; i++){
           tmp = *(mat + row*i + thread_idx)
           tmp *= scale;
           tmp += min_;
           *(mat + row*i + thread_idx) = (T)tmp;

       }

    }

}

template<class T> int
minmax_scale_cpu(T *mat, unsigned int col, unsigned int row, T feature_min, T feature_max){

    T *mat_d = NULL;
    size_t size_mat = sizeof(T)*col*row;
    cudaMalloc((void **)&mat_d, size_mat);
    cudaMemcpy(mat_d, mat, size_mat, cudaMemcpyHostToDevice);
 
    T *min_d = NULL:
    size_t size_min = sizeof(T)*col;
    cudaMalloc((void **)&min_d, size_min);
//    cudaMemcpy(min_d, min, size_min, cudaMemcpyHostToDevice);

    T *max_d = NULL;
    size_t size_max = sizeof(T)*col;
    cudaMalloc((void **)&max_d, size_max);
//    cudaMemcpy(max_d, max, size_max, cudaMemcpyHostToDevice);


    // max blockdim is 65536,so max col not bigger than b5536*65536*256.
    int threaddim = 32;
    int blockdim = ceil(sqrt( ceil(col/threaddim) ));
    dim3 grid_size(blockdim, blockdim);
    dim3 block_size(1,threaddim); 

    get_minmax<T><<<grid_size, block_size>>(mat_d, min_d, max_d);

    minmax_scale_cuda<T><<<grid_size, block_size>>>(mat_d, min_d, max_d, 
                                     col, row, feature_min, feature_max);

    
}
