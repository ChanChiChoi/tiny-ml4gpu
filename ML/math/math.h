#pragma once

#include "common/type.h"
void
matrix_mul_cpu(float *Md, u32 Row_Md, u32 Col_Md,
               float *Nd, u32 Row_Nd, u32 Col_Nd,
               float *Pd, u32 Row_Pd, u32 Col_Pd);

void
cov_cpu(float *mat, u32 Row_mat, u32 Col_mat,
        float *mat_cov, u32 Row_mat_cov, u32 Col_mat_cov);
