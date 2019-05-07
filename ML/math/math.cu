#include <assert.h>
#include <math_functions.hpp>
#include "common/type.h"

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

    // if current thread is exceed the Pd matrix, then return void
    if(Row >= Row_Pd ||Col >= Col_pd)
        return ;

    T Pvalue = 0;

    // for cur tx,ty only care cur WIDTH of Pd
    for(u32 m = 0; m < ceil(Col_Pd/TILE_WIDTH); ++m){
        // get the data again and again
        // if cur tx,ty is exceend of Md,Nd, then it should be exit early,
        // so it will not run here
        Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)]
        Nds[ty][tx] = Nd[(m*TILE_HEIGHT + ty)*Width + Col]
        __syncthreads();

       // calc the point
       for(u32 k = 0; k < TILE_WIDTH; ++k){
          Pvalue += Mds[ty][k] * Nds[k][tx];
       __syncthreads();

   }

   // put the result into origin location of Pd
   Pd[Row*Width + Col] = Pvalue;
}
