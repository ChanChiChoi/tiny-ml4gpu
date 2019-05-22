/* borrowde from https://github.com/allentran/pca-magic/blob/master/ppca/_ppca.py
 *
 * */
#include "ml/include/decomposition/kernel_pca.h"
#include "ml/include/math/math.h"
#include "ml/include/preprocessing/preprocess.h"


// param1 = sigma
// param2 = 幂次
KPCA & 
KPCA::fit(Array &matrix){
    /*
 *        raw = data
 *        // find inf val, replace with max val of the whole matrix
 *        raw[np.isinf(raw)] = np.max(raw[np.isfinite(raw)])
 *        // each feature dim has more than min_obs valid rows
 *        valid_series = np.sum(~np.isnan(self.raw), axis=0) >= min_obs
 *       // extra the valid feature dim
 *       data = raw[:, valid_series].copy()
 *
 * */

    float *ptr_device = (float *)matrix->ptr_buf->ptr_device;
    assert(matrix->ptr_buf->ndim == 2);
    size_t rows = matrix->ptr_buf->shape[0];
    size_t cols = matrix->ptr_buf->shape[1];

    
   // 1 - calc kernel matrix, calc distance between each samples with other samples
   // 1.1 - compute  gramm matrix
    size_t Row_gram = rows;
    size_t Col_gram = rows;
    size_t size_gram = Row_gram*Col_gram*sizeof(float);
    float *K = (float *)malloc(size_gram);
    float *K_device = HOST_TO_DEVICE_MALLOC(K, size_gram);
    
    gram_cpu(ptr_device, rows, cols,
            K_device, Row_gram, Col_gram);
   // 1.2 - compute K matrix
   size_t size_mean_by_row = sizeof(float)*1*Col_gram;
   float *mean = (float *)malloc(size_mean_by_row);
   float *mean_device = HOST_TO_DEVICE_MALLOC(mean, size_mean_by_row);

   mean_by_rows_cpu(K_device, mean_device, Row_gram, Col_gram);
 
   
   float *J = (float *)malloc(size_gram);
   float *J_device = HOST_TO_DEVICE_MALLOC(J,size_gram);
   vector_repeat_by_rows_cpu(J_device, Row_gram, Col_gram,
                          mean_device, Col_gram); 

   matrix_mul_cpu(K_device, Row_gram, Col_gram,
                  J_device, Row_gram, Col_gram,
                  K_device, Row_gram, Col_gram,
                  "divide");
    
   float *J_T_device = nullptr;
   float *J_T_device = DEVICE_MALLOC(J_T_device, size_gram);

   matrix_transpose_cpu(J_device, Row_gram, Col_gram,
                        J_T_device, Col_gram, Row_gram);
   
   matrix_mul_cpu(K_device, Row_gram, Col_gram,
                  J_T_device, Col_gram, Row_gram,
                  K_device, Row_gram, Col_gram,
                  "divide");

    
   // 3 - calc eigenval eigenvector of this matrix
   
   // 4 - trans_mat and output transformed input
    

}


Array * 
KPCA::transform(Array &matrix){

}
