#include "custom_nccl.h"
#include <stdio.h>
#include "common.h"
#include "cuda_runtime.h"
#include "mpi.h"

custom_ncclResult_t custom_RecvCopySend(
    const void* sendbuff,
    void* recvbuff,
    size_t size,
    custom_ncclDataType_t datatype,
    int peer,
    custom_ncclComm_t comm,
    cudaStream_t stream
){
    custom_ncclResult_t ret = custom_ncclSuccess;
    return ret;
}
