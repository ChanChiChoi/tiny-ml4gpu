#pragma once

#include "common/common.h"

void
mean_by_rows_cpu(float *mat_device, float *mean_device, u32 rows, u32 cols);
