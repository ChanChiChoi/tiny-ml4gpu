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
#include "common/buffer_info.h"
#include "common/malloc_free.h"

#define MAX(x,y) ((x)>(y) ? (x): (y))
#define MIN(x,y) ((x)>(y) ? (y): (x))


/* get the [n by m] matrix's maxValue vector and minValue vector by col dimension,
means maxVal vector is [1 by m], minVal vector is [1 by m]*/
template<class T> __global__ void
_get_minmax(T *mat, T *min, T *max, unsigned int cols, unsigned int rows, T min_val, T max_val){

    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int thread_idx = (gridDim.x * blockDim.x)*idy +idx;

    // launch col numbers cuda threads to handle the matrix.
    // each cuda thread handle one col dimension
    if (thread_idx < cols){
        T min_l = max_val;
        T max_l = min_val;
    
        T tmp = (T)0;
        for (unsigned int i = 0; i < rows; i++){
            tmp = *(mat + cols*i + thread_idx);
            min_l = MIN(tmp, min_l);
            max_l = MAX(tmp, max_l);
        } 
        min[thread_idx] = min_l;
        max[thread_idx] = max_l;
    }
}


template<typename T> vector<T *>
get_minmax(Buf &buf){
    
    assert(buf.ptr_device != NULL);

    auto cols = buf.rows_cols()[1];
    size_t size_min = cols * buf.itemsize;
    size_t size_max = size_min;
    
    float *min_d = device_malloc<float>(size_min);
    float *max_d = device_malloc<float>(size_max);

    float min_val = std::numeric_limits<float>::min();
    float max_val = std::numeric_limits<float>::max();

    _get_minmax<T><<<grid_size, block_size>>>((float *)buf.ptr_device, min_d, max_d, cols, rows, min_val, max_val);

    


}


/*normaliza the [n by m] matrix, use col cuda threads*/
template<class T> __global__ void
minmax_scale_cuda(T *mat, T *min, T *max, unsigned int cols, unsigned int rows, T feature_min, T feature_max){
/* feature_range should bigger than 0*/

    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int thread_idx = (gridDim.x * blockDim.x)*idy +idx;

    if (thread_idx < cols){
       T min_l = min[thread_idx];
       T max_l = max[thread_idx];

       T range = max_l - min_l;
       
       T feature_range_l = feature_max - feature_min; // should not be 0, if it's 0, then replace with 1
       assert(feature_range_l > 0 );
       float scale = feature_range_l / range; 
       float min_  = feature_min - min_l * scale;
       
       T tmp = (T)0;
       for(unsigned int i = 0; i<rows; i++){
           tmp = *(mat + cols*i + thread_idx);
           tmp *= scale;
           tmp += min_;
           *(mat + cols*i + thread_idx) = (T)tmp;

       }

    }

}


//TODO: need extra min_d  and max_d.
//TODO: create minmax_scale class

template<class T> void
_minmax_scale_cpu(Buf &buf, unsigned int rows, unsigned int cols, T feature_min, T feature_max){

    size_t size_min = buf.itemsize * cols;
    size_t size_max = buf.itemsize * cols;

    // max blockdim is 65536,so max col not bigger than 65536*65536*256.
    unsigned int threaddim = 32;
    int blockdim = MAX(ceil(sqrt( ceil(cols/threaddim) )),1);
    dim3 grid_size(blockdim, blockdim);
    dim3 block_size(1,threaddim);


    T *min_d = device_malloc<T>(size_min);
    T *max_d = device_malloc<T>(size_max);

    T min_val = std::numeric_limits<T>::min();
    T max_val = std::numeric_limits<T>::max();

    get_minmax<T><<<grid_size, block_size>>>((T *)buf.ptr_device, min_d, max_d, cols, rows, min_val, max_val);
    cudaDeviceSynchronize();
    minmax_scale_cuda<T><<<grid_size, block_size>>>((T *)buf.ptr_device, min_d, max_d,
                                                     cols, rows, feature_min, feature_max);

    device_free(min_d);
    device_free(max_d);
}


int
minmax_scale_cpu(Buf &buf, float feature_min, float feature_max){

    ssize_t ndim = buf.ndim;
    ssize_t rows, cols;
    auto shape = buf.shape;
    auto tmp = buf.rows_cols();
    auto rows = tmp[0];
    auto cols  = tmp[1];

    switch (buf.dtype){
        case Dtype::INT:
            _minmax_scale_cpu<int>(buf, rows, cols, (int)feature_min, (int)feature_max);
            break;
        case Dtype::FLOAT:
            _minmax_scale_cpu<float>(buf, rows, cols, (float)feature_min, (float)feature_max);
            break;

    }
    return 0;
}

//
int
main(){

    float mat[4][2] = {{-1,2},{-0.5,6},{0,10},{1,18}};

    Buf data_buf = Buf( &mat[0][0], 
                   sizeof(float), 
                   Dtype::FLOAT, 
                   2, 
                   {4,2},
                   {2,1}
                   );

    // 1 - copy to device
    host_to_device(data_buf);
    // 2 - get two min max vector
    min_max_vec = get_minmax(buf);
    // 3 - get minmax_scale for train


    // 4 - copy test to device
    // 5 - use two min max vector 
    // 6 - get minmax_scale for test

    // 7 - get result of knn
    minmax_scale_cpu(data_buf, 3, 6);
    device_to_host(data_buf);


    //=================
    from ml4gpu import knn
    knn = KNN()
    knn.fit(train,labels)
    knn.predict(test)
    
    for (int i=0; i<4; i++){
        for (int j=0; j<2; j++){
           printf("%d %d vale %f\n",i,j,mat[i][j]);
        }
    }
}
