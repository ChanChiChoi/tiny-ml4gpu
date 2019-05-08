#include <assert.h>
#include <stdio.h>
#include <math_functions.hpp>
#include "common/type.h"
#include "common/common.h"
#include "common/malloc_free.h"

# define TILE_HEIGHT 32
# define TILE_WIDTH 32

template<typename T> __global__ void
matrix_mul(T * Md, u32 Row_Md, u32 Col_Md,
           T * Nd, u32 Row_Nd, u32 Col_Nd,
           T * Pd, u32 Row_Pd, u32 Col_Pd){
    
    // row = height = x;   
    // col = width = y
    assert(Col_Md == Row_Nd);

    __shared__ T Mds[TILE_HEIGHT][TILE_WIDTH];
    __shared__ T Nds[TILE_HEIGHT][TILE_WIDTH];

    u32 bx = blockIdx.x;
    u32 by = blockIdx.y;
    u32 tx = threadIdx.x;
    u32 ty = threadIdx.y;

    // split Md by TILE_WIDTH, so each blocksize equal TILE_WIDTH
    // current we create the Row,Col in Pd
    u32 Row = by*TILE_HEIGHT + ty;
    u32 Col = bx*TILE_WIDTH + tx;
    
    //printf("[bx by tx ty row col][%d %d %d %d %d %d]\n",bx,by,tx,ty,Row,Col);
    // if current thread is exceed the Pd matrix, then return void
    if(Row >= Row_Pd ||Col >= Col_Pd)
        return ;

    T Pvalue = 0;

    
    // for cur tx,ty only care Col of Md and Row of Nd
    for(u32 m = 0; m < ceil((double)Col_Md/TILE_WIDTH); ++m){
        // get the data again and again
        // if cur tx,ty is exceend of Md,Nd, then it should be exit early,
        // so it will not run here
        const u32 ind_bef_Md = Row*Col_Md;
        const u32 ind_x_Md = m*TILE_WIDTH + tx;

        const u32 ind_y_Nd = m*TILE_HEIGHT + ty;

        // if cur x is exceed col of md, then skip
        if (ind_x_Md < Col_Md)
            Mds[ty][tx] = Md[ind_bef_Md + ind_x_Md];

        // if cur y is exceed row of nd, then skip
        if (ind_y_Nd  < Row_Nd)
            Nds[ty][tx] = Nd[ind_y_Nd*Col_Nd + Col];

        __syncthreads();

       u32 ind_max_TILE;
       if ((m+1)*TILE_WIDTH <= Col_Md)
           ind_max_TILE = TILE_WIDTH;
       else
           ind_max_TILE = Col_Md - m*TILE_WIDTH;
       // calc the point
       for(u32 k = 0; k < ind_max_TILE; ++k)
          Pvalue += Mds[ty][k] * Nds[k][tx];

       __syncthreads();

   } 

   // put the result into origin location of Pd
   Pd[Row*Col_Pd + Col] = Pvalue;
}

template<typename T> void
matrix_mul_launch(T * Md, u32 Row_Md, u32 Col_Md,
           T * Nd, u32 Row_Nd, u32 Col_Nd,
           T * Pd, u32 Row_Pd, u32 Col_Pd){

    dim3 grid(MAX(1, (size_t)ceil((double)Col_Pd/TILE_HEIGHT)),
              MAX(1, (size_t)ceil((double)Row_Pd/TILE_WIDTH)) );
    dim3 block(TILE_WIDTH, TILE_HEIGHT);

    matrix_mul<T><<<grid, block>>>(Md, Row_Md, Col_Md,
           Nd, Row_Nd, Col_Nd,
           Pd, Row_Pd, Col_Pd);
}

void
matrix_mul_cpu(float *Md, u32 Row_Md, u32 Col_Md,
               float *Nd, u32 Row_Nd, u32 Col_Nd,
               float *Pd, u32 Row_Pd, u32 Col_Pd){

    matrix_mul_launch<float>(Md, Row_Md, Col_Md,
               Nd, Row_Nd, Col_Nd,
               Pd, Row_Pd, Col_Pd);
}

int
main(){
//2wx2w × 2wx2w，on p40，need 15s; on e5-2630（2.2ghz）×40core，need 40s
   size_t rowm = 29,colm = 17;
   size_t rown = colm, coln = 17;
   size_t rowp = rowm, colp = coln;
   
   size_t size = sizeof(float)*rowm*colm;
   float *md = (float *)malloc(size);
   for(int i=0;i<rowm*colm;i++)
     md[i] = i;
   float *md_d = host_to_device_malloc(md, size);

   size_t size1 = sizeof(float)*rown*coln;
   float *nd = (float *)malloc(size1);
   for(int i=0;i<rown*coln;i++)
      nd[i]=2;
   float *nd_d = host_to_device_malloc(nd, size1);
   
   size_t size2 = sizeof(float)*rowp*colp;
   float *pd = (float *)malloc(size2);
   float *pd_d = host_to_device_malloc(pd, size2);


   matrix_mul_cpu(md_d,rowm,colm,
                       nd_d,rown,coln,
                       pd_d,rowp,colp);

   device_to_host_free(pd,pd_d,size2);
   for(int i = 0; i<rowp; i++){
      printf("[%d] ",i+1);
      for(int j=0;j<colp;j++)
          printf("%d ",(int)pd[i*colp+j]);
      printf("\n");
   }
  return 0;
}
