#pragma once

void
matrix_subblock_cpu(T *big, u32 Row_big, u32 Col_big,
                T *small, u32 Row_sm, u32 Col_sm,
                u32 rmin, u32 cmin, u32 rmax, u32 cmax);

void
matrix_mul_cpu(float *Md, u32 Row_Md, u32 Col_Md,
               float *Nd, u32 Row_Nd, u32 Col_Nd,
               float *Pd, u32 Row_Pd, u32 Col_Pd);

void
cov_cpu(float *mat, u32 Row_mat, u32 Col_mat,
        float *mat_cov, u32 Row_mat_cov, u32 Col_mat_cov);
