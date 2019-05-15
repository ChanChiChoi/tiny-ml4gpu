#include<assert>
#include<cmath>

#include "common/common.h"
#include "ML/knn/matrix.cuh"

__global__ void
matrix_mul(float *train, size_t train_rows, size_t train_cols,
           float *test, size_t test_rows, size_t test_cols,
           float *ans, size_t ans_rows, size_t ans_cols){

    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  //  unsigned int thread_idx = (gridDim.x * blockDim.x)*idy +idx;

    assert(test_cols == train_cols);

    if (idx < ans_rows && idy < ans_cols){
        float tmp = 0;
    
        // ans [100*32+threadx][100*32+thready]
        for(int i = 0; i < train_cols; i++){
            float x = train[idx*train_cols + i];
            float y = test[idy*test_cols + i];
            tmp += x * y;
    
        }
        ans[idy*train_rows+idx] = tmp; 
    }

}

void
matrix_mul_cpu(float *train, size_t train_rows, size_t train_cols,
           float *test, size_t test_rows, size_t test_cols,
           float *ans, size_t ans_rows, size_t ans_cols){
    // handle by ans matrix, when calc the threads number
    // ans[100*32+]

    unsigned int TILE_WIDTH = 32;
    unsigned int TILE_HEIGHT = 32;
    
    int gx = MAX(1,ceil(train_rows/TILE_WIDTH));//200w
    int gy = MAX(1,ceil(test_rows/TILE_HEIGHT));//30
    dim3 grid(gx,gy);
    dim3 block(TILE_WIDTH,TILE_HEIGHT);

    matrix_mul<<<grid, block>>>(float *train, size_t train_rows, size_t train_cols,
           float *test, size_t test_rows, size_t test_cols,
           float *ans, size_t ans_rows, size_t ans_cols);


}


