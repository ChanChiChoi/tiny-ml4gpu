/* borrowde from https://github.com/allentran/pca-magic/blob/master/ppca/_ppca.py
 *
 * */
#include <string>
#include "ml/include/decomposition/kernel_pca.h"
#include "ml/include/math/math.h"
#include "ml/include/preprocessing/preprocess.h"
#include "common/include/type.h"


// param1 = sigma
// param2 = 幂次
Array * 
KPCA::fit(Array &matrix){
    /* this implementation borrowed from:
 * https://github.com/UMD-ISL/Matlab-Toolbox-for-Dimensionality-Reduction/blob/master/techniques/kernel_pca.m
 * */

    float *ptr_device = (float *)matrix->ptr_buf->ptr_device;
    assert(matrix->ptr_buf->ndim == 2);
    size_t rows = matrix->ptr_buf->shape[0];
    size_t cols = matrix->ptr_buf->shape[1];
    
    /* 1 - calc kernel matrix, calc distance between each samples with other samples */
    // 1.1 - compute gramm matrix
    size_t Row_gram = rows, Col_gram = rows;
    size_t size_gram = Row_gram*Col_gram*sizeof(float);
    float *K_device = nullptr;
    float *K_device = DEVICE_MALLOC(K_device, size_gram);
    
    gram_cpu(ptr_device, rows, cols,
            K_device, Row_gram, Col_gram);

    // 1.2 - compute K matrix
    size_t size_mean_by_row = sizeof(float)*1*Col_gram;
    float *mean = (float *)malloc(size_mean_by_row);
    float *mean_device = HOST_TO_DEVICE_MALLOC(mean, size_mean_by_row);
 
    delete this->column_sums;
    this->column_sums = new Array{
               nullptr, mean, mean_device,
               2,{ssize_t(1), ssize_t(Col_gram)}, std::string(1,'f'),
               ssize_t(sizeof(float)), ssize_t(1*Col_gram),
               {ssize_t(sizeof(float)*Col_gram), ssize_t(sizeof(float))}
               };
    // compute mean by row
    mean_by_rows_cpu(K_device, mean_device, Row_gram, Col_gram);
  
    // calc mean of mean device on host side, then K+ mean
    DEVICE_TO_HOST(mean, mean_device, size_mean_by_row); 
    // computer total_sum
    this->total_sum = 0;
    for (int i=0;i<Col_gram; i++){
         this->total_sum += mean[i];
    }
    this->total_sum /= Col_gram;

    // repeat mean vector to a whole matrix , then K - J
    float *J_device = nullptr;
    float *J_device = DEVICE_MALLOC(J_device, size_gram);
    vector_repeat_by_rows_cpu(J_device, Row_gram, Col_gram,
                           mean_device, Col_gram); 
 
    matrix_mul_cpu(K_device, Row_gram, Col_gram,
                   J_device, Row_gram, Col_gram,
                   K_device, Row_gram, Col_gram,
                   "divide");
    // tranpose J_device, then K - J'
    float *J_T_device = nullptr;
    float *J_T_device = DEVICE_MALLOC(J_T_device, size_gram);
 
    matrix_transpose_cpu(J_device, Row_gram, Col_gram,
                         J_T_device, Col_gram, Row_gram);
    DEVICE_FREE(J_device);
    
    matrix_mul_cpu(K_device, Row_gram, Col_gram,
                   J_T_device, Col_gram, Row_gram,
                   K_device, Row_gram, Col_gram,
                   "divide");
    DEVICE_FREE(J_T_device);
 
    matrix_add_scalar_cpu(K_device, Row_gram, Col_gram, this->total_sum);
 
    // 3 - calc eigenval eigenvector of this matrix
    ci32 Row_A = Row_gram, Col_A = Col_gram;
    ci32 lda = Row_gram;
    ci32 Row_U = lda, Col_U = lda;
    ci32 Length = Col_A;
    ci32 Row_VT = Col_A, Col_VT = Col_A;
    ci32 Row_V = Col_VT, Col_V = Row_VT;
 
    size_t size_U = sizeof(float)*Row_U*Col_U;
    size_t size_S = sizeof(float)*Length;
    size_t size_VT = sizeof(float)*Row_VT*Col_VT;

    float *U = (float *)malloc(size_U);
    float *S = (float *)malloc(size_S);
    float *V = (float *)malloc(size_VT); //for V not VT
    
    float *U_device = HOST_TO_DEVICE_MALLOC(U, size_U);
    float *S_device = HOST_TO_DEVICE_MALLOC(S, size_S);
    float *VT_device = HOST_TO_DEVICE_MALLOC(V, size_VT);
 
    // TODO: K(isnan(K)) = 0;
    //       K(isinf(K)) = 0;
    svd(K_device, Row_gram, Col_gram, lda,
         U_device, Row_U, Col_U,
         S_device, Length,
         VT_device, Row_VT, Col_VT);

    DEVICE_FREE(K_device);
    DEVICE_FREE(U_device);
    free(U):
 
    // 4 - trans_mat and output transformed input
 
    // handle L
    vector_sqrt_self_cpu(S_device, Length);

    delete this->L;
    this->L = new Array{
          nullptr, S, S_device,
          2, {1, ssize_t(Length)}, std::string(1,'f'),
          ssize_t(sizeof(float)), ssize_t(1*Length),
          {ssize_t(sizeof(float)*Length), ssize_t(ssizeof(float))}
        };
    
    float *sqrtL = nullptr;
    size_t size_sqrtL = sizeof(float)*n_components*n_components;
    sqrtL = DEVICE_MALLOC(sqrtL, size_sqrtL);
    matrix_diag_cpu(sqrtL, n_components, n_components, 
                    S_device, n_components);
    // get V from VT
//    float *V_device = nullptr;
//    V_device = DEVICE_MALLOC(V_device, size_VT);
//    matrix_transpose_cpu(VT_device, Row_VT, Col_VT,
//                         V_device, Col_VT, Row_VT);
    // get submatrix of V_T
    float *subVT_device = nullptr;
    size_t Row_subVT = n_components, Col_subV = Col_VT;
    size_t size_subVT = sizeof(float)*Row_subVT*Col_subVT;
    subVT_device = DEVICE_MALLOC(subVT_device, size_subVT);

    matrix_subblock_cpu(VT_device, Row_VT, Col_VT,
                       subVT_device, Row_subVT, Col_subVT,
                       0, 0, Row_subVT, Col_subVT);

    delete this->V_T;
    this->V_T = new Array{

        };
//    float *V_device = nullptr;
//    V_device = DEVICE_MALLOC(V_device, size_VT);
//    matrix_transpose_cpu(VT_device, Row_VT, Col_VT,
//                         V_device, Col_VT, Row_VT);
    
 
}


Array * 
KPCA::transform(Array &matrix){

}


/*
    if ~exist('no_dims', 'var')
        no_dims = 2;
    end
    kernel = 'gauss';
    param1 = 1;
    param2 = 3;
    if nargin > 2
        kernel = varargin{1};
        if length(varargin) > 1 & strcmp(class(varargin{2}), 'double'), param1 = varargin{2}; end
        if length(varargin) > 2 & strcmp(class(varargin{3}), 'double'), param2 = varargin{3}; end
    end

    % Store the number of training and test points
    ell = size(X, 1);

    if size(X, 1) < 2000

        % Compute Gram matrix for training points
        disp('Computing kernel matrix...');
        K = gram(X, X, kernel, param1, param2);

        % Normalize kernel matrix K
        mapping.column_sums = sum(K) / ell;                       % column sums
        mapping.total_sum   = sum(mapping.column_sums) / ell;     % total sum
        J = ones(ell, 1) * mapping.column_sums;                   % column sums (in matrix)
        K = K - J - J';
        K = K + mapping.total_sum;

        % Compute first no_dims eigenvectors and store these in V, store corresponding eigenvalues in L
        disp('Eigenanalysis of kernel matrix...');
        K(isnan(K)) = 0;
        K(isinf(K)) = 0;
        [V, L] = eig(K);
    else
        % Compute column sums (for out-of-sample extension)
        mapping.column_sums = kernel_function([], X', 1, kernel, param1, param2, 'ColumnSums') / ell;
        mapping.total_sum   = sum(mapping.column_sums) / ell;

        % Perform eigenanalysis of kernel matrix without explicitly
        % computing it
        disp('Eigenanalysis of kernel matrix (using slower but memory-conservative implementation)...');
        options.disp = 0;
        options.isreal = 1;
        options.issym = 1;
        [V, L] = eigs(@(v)kernel_function(v, X', 1, kernel, param1, param2, 'Normal'), size(X, 1), no_dims, 'LM', options);
        disp(' ');
    end 

    % Sort eigenvalues and eigenvectors in descending order
    [L, ind] = sort(diag(L), 'descend');
    L = L(1:no_dims);
    V = V(:,ind(1:no_dims));

    % Compute inverse of eigenvalues matrix L
    disp('Computing final embedding...');
    invL = diag(1 ./ L);

    % Compute square root of eigenvalues matrix L
    sqrtL = diag(sqrt(L));

    % Compute inverse of square root of eigenvalues matrix L
    invsqrtL = diag(1 ./ diag(sqrtL));

    % Compute the new embedded points for both K and Ktest-data
    mappedX = sqrtL * V';                     % = invsqrtL * V'* K

    % Set feature vectors in original format
    mappedX = mappedX';

    % Store information for out-of-sample extension
    mapping.X = X;
    mapping.V = V;
    mapping.invsqrtL = invsqrtL;
    mapping.kernel = kernel;
    mapping.param1 = param1;
    mapping.param2 = param2;

 * */
