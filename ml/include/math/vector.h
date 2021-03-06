#pragma once
#include "common/include/type.h"
void
vector_repeat_by_rows_cpu(float *mat_device, u32 rows_mat, u32 cols_mat,
                          float *vector_device, u32 cols_vec);
void
vector_invsqrt_self_cpu(float *vec, u32 len);

void
vector_sqrt_self_cpu(float *vec, u32 len);
