#include "custom_nccl.h"
#include "nccl.h"
#include <stdio.h>
#include "common.h"
#include "cuda_runtime.h"
#include "mpi.h"

custom_ncclResult_t custom_ncclRecv(
    void* recvbuff,
    size_t count,
    custom_ncclDataType_t datatype,
    int peer,
    custom_ncclComm_t comm,
    cudaStream_t stream
){
    ncclDataType_t nccl_datatype = ncclDataType_t(int(datatype));
    ncclComm_t nccl_comm = comm;
    custom_ncclResult_t res = CUSTOMNCCLCHECK(
        ncclRecv(recvbuff, count, nccl_datatype, peer, nccl_comm, stream)
    );
    return res;
}