// Code is borrowed and edited from NVIDIA/NCCL repo, whose copyright is below.
/*************************************************************************
* Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
*
* See LICENSE.txt for license information
************************************************************************/

#ifndef CUSTOM_NCCL_H_
#define CUSTOM_NCCL_H_

#include <cstddef>
#include "nccl.h"  // for bootstrapping custom nccl, will remove as a dependency slowly.
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

/* Error type */
typedef enum { custom_ncclSuccess                 =  0,
    custom_ncclUnhandledCudaError      =  1,
    custom_ncclSystemError             =  2,
    custom_ncclInternalError           =  3,
    custom_ncclInvalidArgument         =  4,
    custom_ncclInvalidUsage            =  5,
    custom_ncclRemoteError             =  6,
    custom_ncclInProgress              =  7,
    custom_ncclNumResults              =  8 } custom_ncclResult_t;

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

// p2p
ncclResult_t custom_RecvSend(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t custom_AllToAll();
ncclResult_t custom_NeighborExchange();
ncclResult_t custom_Gather();
ncclResult_t custom_Satter();
ncclResult_t custom_RecvCopySend(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t custom_RecvReduceCopySend();
ncclResult_t custom_RecvReduceSend();

// collectives
ncclResult_t custom_Broadcast();
ncclResult_t custom_AllReduce();
ncclResult_t custom_ReduceScatter();
ncclResult_t custom_AllGather();
ncclResult_t custom_Reduce();

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
