#pragma once

#include "NVIDIA/common/inc/helper_cuda.h"

#define CHECK_CALL( err, file, line ) check((err), #err, file, line)

#define LAST_ERROR(msg, file, line)  __getLastCudaError(msg, file, line)
