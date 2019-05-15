//#include "knn.h"
#include<time.h>
#include<stdio.h>
#include<cuda_runtime.h>

/*
TODO: add 'dim3 block', 'dim3 thread' check
TODO: add __shared__  size check
TODO: add CHECK_ERROR
TODO: current only support two-dim of dim3

*/

template<class T>
__global__ void neighbors(T *x, T *dataset,const unsigned int col, unsigned int num_samples, T *ans );


//__constant__ float d_x[1000];

template<class T>
int
vec_mul_mat(T *x, T *dataset, T *ans, const unsigned int col, unsigned int num_samples, unsigned int k){
   /* 计算一个向量乘以矩阵，如[1x1000  乘以 1000x200_0000] ,则此时返回1x200_0000
   */    
   T *d_x = NULL;
   size_t size1 = sizeof(T)*col;
   cudaMalloc((void **)&d_x,size1);
   cudaMemcpy(d_x, x, size1,cudaMemcpyHostToDevice);

//   cudaMemcpyToSymbol(d_x,x,size1);   

   //TODO: need handle whether the block is smaller than need!
   dim3 block(100,100);
   dim3 thread(1,256);

   T *d_dataset = NULL;
   size_t size2 = sizeof(T)*col*num_samples;
   cudaMalloc((void **)&d_dataset, size2);
   cudaMemcpy(d_dataset, dataset,size2,cudaMemcpyHostToDevice);
   
   T *d_ans = NULL;
   size_t size3 = sizeof(T)*num_samples;
   cudaMalloc((void **)&d_ans, size3);


   neighbors<T><<<block,thread>>>(d_x,d_dataset,col,num_samples,d_ans); 

   cudaDeviceSynchronize();

   cudaMemcpy(ans, d_ans, size3, cudaMemcpyDeviceToHost);

   cudaFree(d_x);
   cudaFree(d_dataset);
   cudaFree(d_ans);

   return 0;

}

// 先写个以行为单位的版本
template<class T>
__global__ void neighbors(T *x, T *dataset,const unsigned int col, unsigned int num_samples, T *ans ){
    /* 计算向量和矩阵的相乘，至于矩阵和矩阵的相乘，后续再实现*/
//    extern __shared__ T x1[];

    
//    for(int i=0;i<col; i++){
//       x1[i]=x[i] ;
//    }
//    __syncthreads();

    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // 当前每一行的网格数量乘以网格一行的数量，为总的一行的量，× 当前线程的列所在位置
    unsigned int thread_idx = (gridDim.x * blockDim.x)*idy +idx;

    if (thread_idx <= num_samples){
        T tmp = (T)0;
    
        T *p = dataset + thread_idx*col;
        for(int i=0; i<col; i++){
            tmp += x[i] * *(p+i);
        }
        ans[thread_idx] = tmp;
    }
}


//   const unsigned int col = 1000;
//   const unsigned int num_samples = 20;
//   float x[col];
//
//   float ans[num_samples];
//   float dataset[num_samples][col];


float *knn_one(float *x, float *dataset, float *ans, 
             const unsigned int col, const unsigned int num_samples){

   float *p = dataset;
   vec_mul_mat<float>(x,p,ans,col,num_samples,col); 
   return ans;

}

//int
//main(){
//   
//   printf("here=================\n");
//
//   for(int i=0; i<col; i++){
//      x[i] = (float)(i+1);
//   }
//
//   for (int i=0; i<num_samples; i++){
//     ans[i] = 0.0;
//     for(int j=0; j<col; j++){
//        dataset[i][j] = 2.0;
//     }
//   }
//   float *p = &dataset[0][0];
//
//
//   clock_t t;
//   t = clock();
//   vec_mul_mat<float>(x,p,ans,col,num_samples,col);
//   t = clock() - t;
//   printf ("%f seconds \n", ((float)t)/CLOCKS_PER_SEC);
//
//
//   for(int j = 0; j<num_samples; j++){
//       printf("cur%d %f\n",j, ans[j]);
//   }
//   return 0;
//}
