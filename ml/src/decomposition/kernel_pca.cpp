/* borrowde from https://github.com/allentran/pca-magic/blob/master/ppca/_ppca.py
 *
 * */
#include <string>
#include "ml/include/decomposition/kernel_pca.h"
#include "ml/include/math/math.h"
#include "ml/include/preprocessing/preprocess.h"
#include "common/include/type.h"
#include "common/include/buffer_info_ex.h"
#include "common/include/malloc_free.h"


// param1 = sigma
// param2 = 幂次
Array * 
KPCA::fit(Array &matrix){
    /* this implementation borrowed from:
 * https://github.com/UMD-ISL/Matlab-Toolbox-for-Dimensionality-Reduction/blob/master/techniques/kernel_pca.m
 * */

    float *ptr_device = (float *)(matrix.ptr_buf->ptr_device);
    assert(matrix.ptr_buf->ndim == 2);
    size_t rows = matrix.ptr_buf->shape[0];
    size_t cols = matrix.ptr_buf->shape[1];
    
    /* 1 - calc kernel matrix, calc distance between each samples with other samples */
    // 1.1 - compute gramm matrix
    float *mat_T_device = nullptr;
    size_t size_mat = sizeof(float)*rows*cols;
    size_t Row_mat_T = cols, Col_mat_T = rows;
    mat_T_device = DEVICE_MALLOC(mat_T_device, size_mat);
    matrix_transpose_cpu(ptr_device, rows, cols,
                  mat_T_device, Row_mat_T, Col_mat_T);    

    size_t Row_gram = rows, Col_gram = rows;
    float *K_device = nullptr;
    size_t size_gram = sizeof(float)*Row_gram*Col_gram;
    K_device = DEVICE_MALLOC(K_device, size_gram);

    gram_cpu(ptr_device, rows, cols,
             mat_T_device, Row_mat_T, Col_mat_T,
             K_device, Row_gram, Col_gram,
             this->param1);

    DEVICE_FREE(mat_T_device);

    // 1.2 - compute K matrix
    size_t size_mean_by_row = sizeof(float)*1*Col_gram;
    float *mean = (float *)malloc(size_mean_by_row);
    float *mean_device = HOST_TO_DEVICE_MALLOC(mean, size_mean_by_row);

    // compute mean by row
    mean_by_rows_cpu(K_device, mean_device, Row_gram, Col_gram);
  
    // calc mean of mean device on host side, then K+ mean
    DEVICE_TO_HOST(mean, mean_device, size_mean_by_row); 
 
    delete this->column_sums;
    this->column_sums = new Array{
               nullptr, mean, mean_device,
               2,{ssize_t(1), ssize_t(Col_gram)}, std::string(1,'f'),
               ssize_t(sizeof(float)), ssize_t(1*Col_gram),
               {ssize_t(sizeof(float)*Col_gram), ssize_t(sizeof(float))}
               };

    // computer total_sum
    this->total_sum = 0;
    for (int i=0;i<Col_gram; i++){
         this->total_sum += mean[i];
    }
    this->total_sum /= Col_gram;

    // repeat mean vector to a whole matrix , then K - J
    float *J_device = nullptr;
    J_device = DEVICE_MALLOC(J_device, size_gram);
    vector_repeat_by_rows_cpu(J_device, Row_gram, Col_gram,
                           mean_device, Col_gram); 
 
    matrix_mul_cpu(K_device, Row_gram, Col_gram,
                   J_device, Row_gram, Col_gram,
                   K_device, Row_gram, Col_gram,
                   SCALAR_TWO_DIVIDE);
    // tranpose J_device, then K - J'
    float *J_T_device = nullptr;
    J_T_device = DEVICE_MALLOC(J_T_device, size_gram);
 
    matrix_transpose_cpu(J_device, Row_gram, Col_gram,
                         J_T_device, Col_gram, Row_gram);
    DEVICE_FREE(J_device);
    
    matrix_mul_cpu(K_device, Row_gram, Col_gram,
                   J_T_device, Col_gram, Row_gram,
                   K_device, Row_gram, Col_gram,
                   SCALAR_TWO_DIVIDE);
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
    free(U);
 
    // 4 - trans_mat and output transformed input
 
    // handle L
    float *L_device = nullptr;
    L_device = DEVICE_MALLOC(L_device, size_S);
    matrix_subblock_cpu(S_device, 1, Length,
                        L_device, 1, Length,
                        0,0,1,Length);
    vector_sqrt_self_cpu(L_device, Length);

    delete this->L;
    this->L = new Array{
          nullptr, S, S_device,
          2, {1, ssize_t(Length)}, std::string(1,'f'),
          ssize_t(sizeof(float)), ssize_t(1*Length),
          {ssize_t(sizeof(float)*Length), ssize_t(sizeof(float))}
        };
    
    float *sqrtL_device = nullptr;
    size_t Row_sqrtL = n_components, Col_sqrtL = n_components;
    size_t size_sqrtL = sizeof(float)*n_components*n_components;
    sqrtL_device = DEVICE_MALLOC(sqrtL_device, size_sqrtL);
    matrix_diag_cpu(sqrtL_device, n_components, n_components,
                    L_device, n_components);

    DEVICE_FREE(L_device);
    // get V from VT
//    float *V_device = nullptr;
//    V_device = DEVICE_MALLOC(V_device, size_VT);
//    matrix_transpose_cpu(VT_device, Row_VT, Col_VT,
//                         V_device, Col_VT, Row_VT);
    // get submatrix of V_T
    float *subVT_device = nullptr;
    size_t Row_subVT = n_components, Col_subVT = Col_VT;
    size_t size_subVT = sizeof(float)*Row_subVT*Col_subVT;
    subVT_device = DEVICE_MALLOC(subVT_device, size_subVT);

    matrix_subblock_cpu(VT_device, Row_VT, Col_VT,
                       subVT_device, Row_subVT, Col_subVT,
                       0, 0, Row_subVT, Col_subVT);

    delete this->V_T;
    this->V_T = new Array{
          nullptr, nullptr, subVT_device,
          2, {ssize_t(Row_subVT), ssize_t(Col_subVT)}, std::string(1,'f'),
          ssize_t(sizeof(float)), ssize_t(Row_subVT*Col_subVT),
          {ssize_t(sizeof(float)*Row_subVT), ssize_t(sizeof(float))}
        };
    
    //mappedX =  sqrtL* subVT
    float *mappedX_device = nullptr;
    size_t Row_mappedX = Row_sqrtL, Col_mappedX = Col_subVT;
    size_t size_mappedX = sizeof(float)*Row_mappedX*Col_mappedX;
    mappedX_device = DEVICE_MALLOC(mappedX_device, size_mappedX);

    matrix_dotmul_cpu(sqrtL_device, Row_sqrtL, Col_sqrtL,
                      subVT_device, Row_subVT, Col_subVT,
                      mappedX_device, Row_mappedX, Col_mappedX);

    DEVICE_FREE(sqrtL_device);

    // get mappedX'
    float *mappedX_T_device = nullptr;
    size_t Row_mappedX_T = Col_mappedX,       
           Col_mappedX_T = Row_mappedX;
    size_t size_mappedX_T = sizeof(float)*Row_mappedX_T*Col_mappedX_T;
    mappedX_T_device = DEVICE_MALLOC(mappedX_T_device, size_mappedX_T);

    matrix_transpose_cpu(mappedX_device, Row_mappedX, Col_mappedX,
                         mappedX_T_device, Row_mappedX_T, Col_mappedX_T);
   
    DEVICE_FREE(mappedX_device);
    
    return new Array{
          nullptr, nullptr, mappedX_T_device,
          2, {ssize_t(Row_mappedX_T), ssize_t(Col_mappedX_T)}, std::string(1,'f'),
          ssize_t(sizeof(float)), ssize_t(Row_mappedX_T*Col_mappedX_T),
          {ssize_t(sizeof(float)*Row_mappedX_T), ssize_t(sizeof(float))}
        };


//    float *V_device = nullptr;
//    V_device = DEVICE_MALLOC(V_device, size_VT);
//    matrix_transpose_cpu(VT_device, Row_VT, Col_VT,
//                         V_device, Col_VT, Row_VT);
    
 
}


Array * 
KPCA::transform(Array &train, Array &test){
/*
 * % Compute and center kernel matrix
 * K = gram(mapping.X, point, mapping.kernel, mapping.param1, mapping.param2);
 * J = repmat(mapping.column_sums', [1 size(K, 2)]);
 * K = K - repmat(sum(K, 1), [size(K, 1) 1]) - J + repmat(mapping.total_sum, [size(K, 1) size(K, 2)]);
 *                                                 
 * % Compute transformed points
 * t_point = mapping.invsqrtL * mapping.V' * K;
 * t_point = t_point';
 *
 * */
    //1 - gram

    float *X_device = (float *)train.ptr_buf->ptr_device;
    float *point_device = (float *)test.ptr_buf->ptr_device;
    // get point_T_device

    size_t Row_X = train.ptr_buf->shape[0];
    size_t Col_X = train.ptr_buf->shape[1];

    size_t Row_point = test.ptr_buf->shape[0];
    size_t Col_point = test.ptr_buf->shape[1];


    float *point_T_device = nullptr;
    size_t Row_point_T = Col_point, Col_point_T = Row_point;
    size_t size_point = sizeof(float)*Row_point*Col_point;
    point_T_device = DEVICE_MALLOC(point_T_device, size_point);    
    matrix_transpose_cpu(point_device, Row_point, Col_point,
                         point_T_device, Row_point_T, Col_point_T);

    float *K_device = nullptr;
    size_t Row_K = Row_X, Col_K = Col_point_T;
    size_t size_K = sizeof(float)*Row_K*Col_K;
    K_device = DEVICE_MALLOC(K_device, size_K);
    // get K
    gram_cpu(X_device, Row_X, Col_X,
             point_T_device, Row_point_T, Col_point_T,
             K_device, Row_K, Col_K,
             this->param1);

    DEVICE_FREE(point_T_device); 

    //2 - computer K
    //K - mean
    //get mean
    float *mean_device = nullptr;
    size_t Row_mean = 1, Col_mean = Col_K;
    size_t size_mean = sizeof(float)*Row_mean*Col_mean;
    mean_device = DEVICE_MALLOC(mean_device,size_mean);

    // compute mean by row
    mean_by_rows_cpu(K_device, mean_device, Row_K, Col_K);
    
    float *mean_rep_device = nullptr;
    size_t Row_mean_rep = Row_K, Col_mean_rep = Col_K;
    size_t size_mean_rep = sizeof(float)*Row_mean_rep*Col_mean_rep;
    mean_rep_device = DEVICE_MALLOC(mean_rep_device, size_mean_rep);

    vector_repeat_by_rows_cpu(mean_rep_device, Row_mean_rep, Col_mean_rep,
                              mean_device, Col_mean);

    DEVICE_FREE(mean_device);

    matrix_mul_cpu(K_device, Row_K, Col_K,
                   mean_rep_device, Row_mean_rep, Col_mean_rep,
                   K_device, Row_K, Col_K,
                   SCALAR_TWO_DIVIDE);
    DEVICE_FREE(mean_rep_device);

    // K - J begin
    float *J_T_device = nullptr;
    size_t Row_J_T = Row_K, Col_J_T = Col_K;
    size_t size_J_T = sizeof(float)*Row_J_T*Col_J_T;    
    J_T_device = DEVICE_MALLOC(J_T_device, size_J_T);
    
    vector_repeat_by_rows_cpu(J_T_device, Row_J_T, Col_J_T,
                         (float *)this->column_sums->ptr_buf->ptr_device, this->column_sums->ptr_buf->shape[1] );
    // transpose J
    float *J_device = nullptr;
    size_t Row_J = Col_J_T, Col_J = Row_J_T;
    size_t size_J = size_J_T;
    J_device = DEVICE_MALLOC(J_device, size_J);
    matrix_transpose_cpu(J_T_device, Row_J_T, Col_J_T,
                         J_device, Row_J, Col_J);
    DEVICE_FREE(J_T_device);
    // K-J
    matrix_mul_cpu(K_device, Row_K, Col_K,
                   J_device, Row_J, Col_J,
                   K_device, Row_K, Col_K,
                   SCALAR_TWO_DIVIDE);

    DEVICE_FREE(J_device);

    matrix_add_scalar_cpu(K_device, Row_K, Col_K, this->total_sum);
    
    //3 - transform
    // computer V'*K
    float *t_point2_device = nullptr;   
    size_t Row_t_point2 = this->n_components, Col_t_point2 = Row_K;
    size_t size_t_point2 = sizeof(float)*Row_t_point2*Col_t_point2;
    t_point2_device = DEVICE_MALLOC(t_point2_device, size_t_point2);

    matrix_dotmul_cpu((float *)this->V_T->ptr_buf->ptr_device, this->V_T->ptr_buf->shape[0], this->V_T->ptr_buf->shape[1],
                     K_device, Row_K, Col_K,
                     t_point2_device, Row_t_point2, Col_t_point2); 

    DEVICE_FREE(K_device);
    // copy this->L to L_device
    float *L_device = nullptr;
    size_t Row_L = 1, Col_L = this->L->ptr_buf->shape[1];
    L_device = DEVICE_MALLOC(L_device, sizeof(float)*Row_L*Col_L);
    matrix_subblock_cpu((float *)this->L->ptr_buf->ptr_device, Row_L, Col_L,
                        L_device, Row_L, Col_L,
                        0,0,Row_L, Col_L);
    // computer invsqrt
    vector_invsqrt_self_cpu(L_device, Col_L);
    
    // compute invsqrtL *ans
    float *invsqrtL_device = nullptr;
    size_t Row_invsqrtL = n_components, Col_invsqrtL = n_components;
    size_t size_invsqrtL = sizeof(float)*Row_invsqrtL*Col_invsqrtL;
    invsqrtL_device = DEVICE_MALLOC(invsqrtL_device, size_invsqrtL);
    matrix_diag_cpu(invsqrtL_device, Row_invsqrtL, Col_invsqrtL,
                     L_device, Col_L);

    DEVICE_FREE(L_device);

    float *res_device = nullptr;
    size_t Row_res = Row_invsqrtL, Col_res = Col_t_point2;
    res_device = DEVICE_MALLOC(res_device, sizeof(float)*Row_res*Col_res);

    matrix_dotmul_cpu(invsqrtL_device, Row_invsqrtL, Col_invsqrtL,
                      t_point2_device, Row_t_point2, Col_t_point2,
                      res_device, Row_res, Col_res);
    DEVICE_FREE(t_point2_device);
    DEVICE_FREE(invsqrtL_device);

    return new Array{
           nullptr, nullptr, res_device,
           2, {ssize_t(Row_res), ssize_t(Col_res)}, std::string(1,'f'),
           ssize_t(sizeof(float)), ssize_t(Row_res*Col_res),
           {ssize_t(sizeof(float)*Row_res), ssize_t(sizeof(float))}

        };
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
