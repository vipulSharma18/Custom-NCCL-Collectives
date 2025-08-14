#include <stdio.h>
#include "custom_nccl.h"
#include "nccl.h"
#include "common.h"
#include "cuda_runtime.h"
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif

ncclResult_t custom_RecvSend(
    const void* sendbuff,
    void* recvbuff,
    size_t size,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream
    ){
    // p2p calls within a group are independent, so the send and recv is done concurrently.
    ncclGroupStart();
    ncclRecv(recvbuff, size, datatype, peer, comm, stream);
    ncclSend(sendbuff, size, datatype, peer, comm, stream);
    ncclResult_t ret = ncclGroupEnd();
    return ret;
}
