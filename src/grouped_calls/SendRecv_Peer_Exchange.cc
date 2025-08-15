#include <stdio.h>
#include "custom_nccl.h"
#include "common.h"
#include "mpi.h"
#include "cuda_runtime.h"

custom_ncclResult_t custom_SendRecv(
    const void* sendbuff,
    void* recvbuff,
    size_t size,
    custom_ncclDataType_t datatype,
    int peer,
    custom_ncclComm_t comm,
    cudaStream_t stream
    ){
    // p2p calls within a group are independent, so the send and recv is done concurrently.
    // peers exchange data concurrently.
    custom_ncclGroupStart();
    custom_ncclRecv(recvbuff, size, datatype, peer, comm, stream);
    custom_ncclSend(sendbuff, size, datatype, peer, comm, stream);
    custom_ncclResult_t res = custom_ncclGroupEnd(); 
    return res;
}
