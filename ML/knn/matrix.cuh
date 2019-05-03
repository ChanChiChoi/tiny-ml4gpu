#include<cuda_runtime.h>
void
matrix_mul_cpu(float *train, size_t train_rows, size_t train_cols,
           float *test, size_t test_rows, size_t test_cols,
           float *ans, size_t ans_rows, size_t ans_cols);
