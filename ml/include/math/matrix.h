#pragma once
#include "common/include/type.h"
#include "ml/include/scalar_op_def.h"

void
matrix_subblock_cpu(float *big, u32 Row_big, u32 Col_big,
                float *small, u32 Row_sm, u32 Col_sm,
                u32 rmin, u32 cmin, u32 rmax, u32 cmax);

void
matrix_dotmul_cpu(float *Md, u32 Row_Md, u32 Col_Md,
               float *Nd, u32 Row_Nd, u32 Col_Nd,
               float *Pd, u32 Row_Pd, u32 Col_Pd,
               const int op = SCALAR_TWO_MUL);

void
matrix_mul_cpu(float *Md, u32 Row_Md, u32 Col_Md,
               float *Nd, u32 Row_Nd, u32 Col_Nd,
               float *Pd, u32 Row_Pd, u32 Col_Pd,
               const int op = SCALAR_TWO_MUL);
void
matrix_divide_scalar_cpu(float *mat, u32 Row, u32 Col, float scalar);

void
matrix_transpose_cpu(float *mat_src, u32 Row_src, u32 Col_src,
                     float * mat_dst, u32 Row_dst, u32 Col_dst);

void
matrix_scalar_sqrt_cpu(float *mat, u32 Row_mat, u32 Col_mat);

void
matrix_diag_cpu(float *mat, u32 Row, u32 Col, float *vec, u32 len);

void
matrix_add_scalar_cpu(float *mat, u32 Row, u32 Col, float scalar);

void
matrix_gaussian_scalar_cpu(float *mat, u32 Row, u32 Col, float scalar_sigma);

