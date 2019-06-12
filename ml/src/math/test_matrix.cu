#include <stdio.h>
#include <cuda_runtime.h>
//#include <chrono>

#include "ml/include/math/math.h"
#include "common/include/helper.cuh"
//using namespace std::chrono;

/*
nvcc test_matrix.cu -I../../../ -std=c++11 -L. -lm4g_ml_math -lm4g_com_cuMF
*/

void
test_mul(){

    //======md
    size_t md_rows = 4;
    size_t md_cols = 50;
    size_t size = md_rows*md_cols*sizeof(float);
    float *md = (float *)malloc(size);
    for(u32 i=0;i<md_cols*md_rows;i++)
        md[i] = i;

    float *md_device;
    cudaMalloc((void **)&md_device,size);
    cudaMemcpy(md_device, md, size, cudaMemcpyHostToDevice);
   //========nd
    size_t nd_rows = 50;
    size_t nd_cols = 4;
    size_t size1 = nd_rows*nd_cols*sizeof(float);
    float *nd = (float *)malloc(size1);
    for(u32 i=0;i<nd_cols*nd_rows; i++){
        nd[i]=2;
    }

    float *nd_device = nullptr;
    cudaMalloc((void **)&nd_device,size1);
    cudaMemcpy(nd_device, nd, size1, cudaMemcpyHostToDevice);
    
    //=======pd
    size_t pd_rows = 4;
    size_t pd_cols = 4;
    size_t size2 = pd_rows*pd_cols*sizeof(float);
    float *pd = (float *)malloc(size2);
    float *pd_device = nullptr;
    cudaMalloc((void **)&pd_device, size2);
    cudaMemcpy(pd_device, pd, size2, cudaMemcpyHostToDevice);
  
    matrix_dotmul_cpu(md_device, md_rows, md_cols,
                   nd_device, nd_rows, nd_cols,
                   pd_device, pd_rows, pd_cols,SCALAR_TWO_MUL);
  //  auto t0 = high_resolution_clock::now();
    //mean_by_rows_cpu(mat_device,mean_device, rows,cols);
    //cudaDeviceSynchronize();

  //  auto t1 = high_resolution_clock::now();

    cudaMemcpy(md, md_device, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(nd, nd_device, size1, cudaMemcpyDeviceToHost);
    cudaMemcpy(pd, pd_device, size2, cudaMemcpyDeviceToHost);

//
    for(u32 i=0; i<4*4;i++){
        printf("val %d %f %f %f\n",i,md[i], nd[i], pd[i]);
    }

    cudaFree(md_device);
    cudaFree(nd_device);
    cudaFree(pd_device);
   // printf("take time %d\n",duration_cast<milliseconds>(t1-t0).count());
    free(md);
    free(nd);
    free(pd);

}

void
test_matrix_sub_by_rows(){

    //======md
    size_t md_rows = 50;
    size_t md_cols = 4;
    size_t size = md_rows*md_cols*sizeof(float);
    float *md = (float *)malloc(size);
    for(u32 i=0;i<md_cols*md_rows;i++)
        md[i] = i;

    float *md_device;
    cudaMalloc((void **)&md_device,size);
    cudaMemcpy(md_device, md, size, cudaMemcpyHostToDevice);
   //========nd
    size_t nd_rows = 3;
    size_t nd_cols = 4;
    size_t size1 = nd_rows*nd_cols*sizeof(float);
    float *nd = (float *)malloc(size1);

    float *nd_device = nullptr;
    cudaMalloc((void **)&nd_device,size1);
    cudaMemcpy(nd_device, nd, size1, cudaMemcpyHostToDevice);
    
    int pd[3] = {1,2,3};
    int length = 3;
    
    //=======pd
    size_t size2 = sizeof(int)*length;
    //float *pd = (float *)malloc(size2);
    int *pd_device = nullptr;
    cudaMalloc((void **)&pd_device, size2);
    cudaMemcpy(pd_device, pd, size2, cudaMemcpyHostToDevice);
  
    matrix_sub_by_rows_cpu(md_device, md_rows, md_cols,
                   nd_device, nd_rows, nd_cols,
                   pd_device, length);

    cudaMemcpy(md, md_device, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(nd, nd_device, size1, cudaMemcpyDeviceToHost);
    cudaMemcpy(pd, pd_device, size2, cudaMemcpyDeviceToHost);

//
    for(u32 i=0; i<12;i++){
        printf("val %d %f %f %d\n",i,md[i], nd[i], pd[i]);
    }

    cudaFree(md_device);
    cudaFree(nd_device);
    cudaFree(pd_device);
   // printf("take time %d\n",duration_cast<milliseconds>(t1-t0).count());
    free(md);
    free(nd);



}
int
main(){
//    test_mul();
    test_matrix_sub_by_rows();
    return 0;
}

