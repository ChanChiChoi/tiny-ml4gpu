#pragma once

#include "common/include/type.h"

void
mean_by_rows_cpu(float *mat_device, float *mean_device, u32 rows, u32 cols);

void
subtract_mean_by_rows_cpu(float *mat_device, float *mean_device, u32 rows, u32 cols);

void
normalization_by_rows_cpu(float *mat_device, float *mean_device, float *std_device, u32 rows, u32 cols);
