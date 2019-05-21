/* borrowde from https://github.com/allentran/pca-magic/blob/master/ppca/_ppca.py
 *
 * */
#include "ml/include/decomposition/kernel_pca.h"
#include "ml/include/math/math.h"


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
    size_t size_gram = rows*rows*sizeof(float);
    float *K = (float *)malloc(size_gram);
    float *K_device = HOST_TO_DEVICE_MALLOC(K, size_gram);
    
    gram_cpu(ptr_device, rows, cols,
            K_device, rows, rows);
   // 1.2 - compute K matrix

   // 3 - calc eigenval eigenvector of this matrix
   
   // 4 - trans_mat and output transformed input
    

}


Array * 
KPCA::transform(Array &matrix){

}
