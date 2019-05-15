#pragma once
#include "common/include/type.h"

void
matrix_subblock_cpu(float *big, u32 Row_big, u32 Col_big,
                float *small, u32 Row_sm, u32 Col_sm,
                u32 rmin, u32 cmin, u32 rmax, u32 cmax);

void
matrix_mul_cpu(float *Md, u32 Row_Md, u32 Col_Md,
               float *Nd, u32 Row_Nd, u32 Col_Nd,
               float *Pd, u32 Row_Pd, u32 Col_Pd);

void
matrix_divide_scalar_cpu(float *mat, u32 Row, u32 Col, u32 scalar);

void
matrix_transpose_cpu(float *mat_src, u32 Row_src, u32 Col_src,
                     float * mat_dst, u32 Row_dst, u32 Col_dst);

void
cov_cpu(float *mat, u32 Row_mat, u32 Col_mat,
        float *mat_cov, u32 Row_mat_cov, u32 Col_mat_cov);
