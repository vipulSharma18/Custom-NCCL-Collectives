#include "custom_nccl.h"
#include "nccl.h"
#include <exception>
#include <stdio.h>
#include "common.h"
#include "cuda_runtime.h"
#include <string>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif

ncclResult_t custom_RecvCopySend(){
    ncclResult_t ret = ncclSuccess;
    return ret;
}
