#include "custom_nccl.h"
#include "nccl.h"
#include <stdio.h>
#include "common.h"
#include "cuda_runtime.h"
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif

ncclResult_t custom_RecvCopySend(
    const void* sendbuff,
    void* recvbuff,
    size_t size,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream
){
    ncclResult_t ret = ncclSuccess;
    return ret;
}
