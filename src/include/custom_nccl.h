// Code is derived from NVIDIA/NCCL repo.
// NVIDIA license is in this repo: https://github.com/NVIDIA/nccl/blob/master/LICENSE.txt
// NVIDIA/NCCL's header code: https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in

#ifndef CUSTOM_NCCL_H_
#define CUSTOM_NCCL_H_

#include "nccl.h"  // for custom_ncclComm typedef only.
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif
#if CUDART_VERSION >= 12080
#include <cuda_fp6.h>
#endif
#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CUSTOM_NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[CUSTOM_NCCL_UNIQUE_ID_BYTES]; } custom_ncclUniqueId;
typedef struct ncclComm custom_ncclComm;
typedef custom_ncclComm* custom_ncclComm_t;

/* Error type */
typedef enum { custom_ncclSuccess      =  0,
    custom_ncclUnhandledCudaError      =  1,
    custom_ncclSystemError             =  2,
    custom_ncclInternalError           =  3,
    custom_ncclInvalidArgument         =  4,
    custom_ncclInvalidUsage            =  5,
    custom_ncclRemoteError             =  6,
    custom_ncclInProgress              =  7,
    custom_ncclNumResults              =  8 
} custom_ncclResult_t;

/* Data types */ //TODO: extend this for fp4
typedef enum {
    custom_ncclInt8       = 0, custom_ncclChar       = 0,
    custom_ncclUint8      = 1,
    custom_ncclInt32      = 2, custom_ncclInt        = 2,
    custom_ncclUint32     = 3,
    custom_ncclInt64      = 4,
    custom_ncclUint64     = 5,
    custom_ncclFloat16    = 6, custom_ncclHalf       = 6,
    custom_ncclFloat32    = 7, custom_ncclFloat      = 7,
    custom_ncclFloat64    = 8, custom_ncclDouble     = 8,
    custom_ncclBfloat16   = 9,
    custom_ncclFloat8e4m3 = 10,
    custom_ncclFloat8e5m2 = 11,
    custom_ncclNumTypes   = 12
} custom_ncclDataType_t;

// nccl_core communicator
custom_ncclResult_t custom_ncclGetUniqueId(custom_ncclUniqueId* uniqueId);
custom_ncclResult_t custom_ncclCommInitRank(custom_ncclComm_t* comm, int nranks, custom_ncclUniqueId commId, int rank);
custom_ncclResult_t custom_ncclCommDestroy(custom_ncclComm_t comm);

// nccl_core group management
custom_ncclResult_t custom_ncclGroupStart();
custom_ncclResult_t custom_ncclGroupEnd();

// nccl_core error management
const char* custom_ncclGetErrorString(custom_ncclResult_t result);
const char* custom_ncclGetLastError(custom_ncclComm_t comm);

// p2p - part of nccl_core
custom_ncclResult_t custom_ncclSend(const void* sendbuff, size_t count,
    custom_ncclDataType_t datatype, int peer, custom_ncclComm_t comm, cudaStream_t stream);
custom_ncclResult_t custom_ncclRecv(void* recvbuff, size_t count,
    custom_ncclDataType_t datatype, int peer, custom_ncclComm_t comm, cudaStream_t stream);

// p2p grouped calls
custom_ncclResult_t custom_SendRecv(const void* sendbuff, void* recvbuff, size_t count,
    custom_ncclDataType_t datatype, int peer, custom_ncclComm_t comm, cudaStream_t stream);
custom_ncclResult_t custom_AllToAll();
custom_ncclResult_t custom_NeighborExchange();
custom_ncclResult_t custom_Gather();
custom_ncclResult_t custom_Satter();
custom_ncclResult_t custom_RecvCopySend();
custom_ncclResult_t custom_RecvReduceCopySend();
custom_ncclResult_t custom_RecvReduceSend();

// collectives
custom_ncclResult_t custom_Broadcast();
custom_ncclResult_t custom_AllReduce();
custom_ncclResult_t custom_ReduceScatter();
custom_ncclResult_t custom_AllGather();
custom_ncclResult_t custom_Reduce();

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
