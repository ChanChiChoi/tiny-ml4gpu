/*
the file imitates sklearn.preprocessing.data
*/
#include<limit>

#define MAX(x,y) ((x)>(y) ? (x): (y))
#define MIN(x,y) ((x)>(y) ? (y): (x))


template<class T> __device__
T * get_minmax(T *, )

// 1 - there is a [n by m] matrix
// 2 - get two vector, one is min, one is max, their size are 1 by m


/* get the [n by m] matrix's maxValue vector and minValue vector by col dimension,
means maxVal vector is [1 by m], minVal vector is [1 by m]*/
template<class T> __global__ void
T * get_minmax(T *mat, T *min, T *max, unsigned int col, unsigned int row){
    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int thread_idx = (gridDim.x * blockDim.x)*idy +idx;

    if (thread_idx >= row){
        T min_l = std::numeric_limits<T>::max();
        T max_l = std::numeric_limits<T>::min();
    
        T tmp = (T)0;
        for (unsigned int i = 0; i < col; i++){
            tmp = *(mat + row*i + thread_idx)
            min_l = MIN(tmp, min_l)
            max_l = MAX(tmp, max_l)
        } 
        min[thread_idx] = min_l;
        max[thread_idx] = max_l;
    }
}





