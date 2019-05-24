#include <assert.h>
#include <stdio.h>
#include "common/include/type.h"
#include "common/include/common.h"
#include "common/include/malloc_free.h"
#include "ml/include/math/math.h"


void
cov_cpu(float *mat, u32 Row_mat, u32 Col_mat,
        float *mat_cov, u32 Row_mat_cov, u32 Col_mat_cov){
    
    //1 - malloc one matrix
    size_t size = sizeof(float)*Row_mat*Col_mat;
    float *mat_T_device = NULL;
    mat_T_device = DEVICE_MALLOC(mat_T_device, size);

    //2 - transpose
    u32 Row_mat_T = Col_mat;
    u32 Col_mat_T = Row_mat;
    matrix_transpose_cpu(mat,Row_mat, Col_mat,
                  mat_T_device, Row_mat_T, Col_mat_T);

    //3 - matrix_mul

    matrix_dotmul_cpu(mat_T_device,Row_mat_T, Col_mat_T,
                   mat, Row_mat, Col_mat,
                   mat_cov, Row_mat_cov, Col_mat_cov,"mul");

    DEVICE_FREE(mat_T_device);

    //4 - divide (n-1) samples;
    size_t n_1 = MAX(1,Row_mat-1);
    matrix_divide_scalar_cpu(mat_cov, Row_mat_cov, Col_mat_cov, n_1);

}

void
gram_cpu(float *mat, u32 Row_mat, u32 Col_mat,
         float mat_T_device, u32 Row_mat_T, u32 Col_mat_T,
         float *mat_gram, u32 Row_gram, u32 Col_gram, const float param1){

    // computer L2_distance  
    matrix_dotmul_cpu(mat,Row_mat, Col_mat,
                   mat_T_device, Row_mat_T, Col_mat_T,
                   mat_gram, Row_gram, Col_gram,"mse");
    
    // computer sqrt of each elements
    matrix_scalar_sqrt_cpu(mat_gram, Row_gram, Col_gram);
    // computer gaussian of each elements
    matrix_gaussian_scalar_cpu(mat_gram, Row_gram, Col_gram, param1);

}


float
sum_vec(float *vec, size_t size){
    float ans = 0;
    for (int i = 0;i<size i++){
        ans += vec[i];
    }
    return ans;

}
