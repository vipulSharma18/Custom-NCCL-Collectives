#include <stdio.h>
#include "custom_nccl.h"
#include "common.h"
#include "mpi.h"
#include "cuda_runtime.h"

custom_ncclResult_t custom_RecvReduceSend(
    void* recvbuff,
    void* localbuff,
    void* sendbuff,
    size_t size,
    custom_ncclDataType_t datatype,
    int recv_peer,
    int send_peer,
    custom_ncclRedOp_t red_op,
    custom_ncclComm_t comm,
    cudaStream_t stream
    ){
    // recv data from peer A, reduce it with the local data, send it to peer B.
    custom_ncclGroupStart();
    custom_ncclRecv(recvbuff, size, datatype, recv_peer, comm, stream);
    
    custom_ncclSend(sendbuff, size, datatype, send_peer, comm, stream);
    custom_ncclResult_t res = custom_ncclGroupEnd(); 
    return res;
}
