/* borrowde from https://github.com/allentran/pca-magic/blob/master/ppca/_ppca.py
 *
 * */
#include "ml/include/decomposition/ppca.h"


PPCA & 
PPCA::fit(Array &matrix){
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

    size_t d = cols;

    // 1 - standardize the data

    // 2 - init necessary variables
    
   // 3 - loop for E step and M step
   //
   // 4 - get C matrix ,then get trans_mat
    

}


Array * 
PPCA::transform(Array &matrix){

}
