/*************************************************************************
* Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
*
* See LICENSE.txt for license information
************************************************************************/

#ifndef NCCL_H_
#define NCCL_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
* Broadcast
*
* Copies count values from root to all other devices.
* root is the rank (not the CUDA device) where data resides before the
* operation is started.
*
* In-place operation will happen if sendbuff == recvbuff.
*/
ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

/*
* All-Reduce
*
* Reduces data arrays of length count in sendbuff using op operation, and
* leaves identical copies of result on each recvbuff.
*
* In-place operation will happen if sendbuff == recvbuff.
*/
ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

/*
* Reduce-Scatter
*
* Reduces data in sendbuff using op operation and leaves reduced result
* scattered over the devices so that recvbuff on rank i will contain the i-th
* block of the result.
* Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
* should have a size of at least nranks*recvcount elements.
*
* In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
*/
ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);

/*
* All-Gather
*
* Each device gathers sendcount values from other GPUs into recvbuff,
* receiving data from rank i at offset i*sendcount.
* Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
* should have a size of at least nranks*sendcount elements.
*
* In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
*/
ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

/*
* Send
*
* Send data from sendbuff to rank peer.
*
* Rank peer needs to call ncclRecv with the same datatype and the same count from this
* rank.
*
* This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
* need to progress concurrently to complete, they must be fused within a ncclGroupStart/
* ncclGroupEnd section.
*/
ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

/*
* Receive
*
* Receive data from rank peer into recvbuff.
*
* Rank peer needs to call ncclSend with the same datatype and the same count to this
* rank.
*
* This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
* need to progress concurrently to complete, they must be fused within a ncclGroupStart/
* ncclGroupEnd section.
*/
ncclResult_t pncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

/*
* Group semantics
*
* When managing multiple GPUs from a single thread, and since NCCL collective
* calls may perform inter-CPU synchronization, we need to "group" calls for
* different ranks/devices into a single call.
*
* Grouping NCCL calls as being part of the same collective operation is done
* using ncclGroupStart and ncclGroupEnd. ncclGroupStart will enqueue all
* collective calls until the ncclGroupEnd call, which will wait for all calls
* to be complete. Note that for collective communication, ncclGroupEnd only
* guarantees that the operations are enqueued on the streams, not that
* the operation is effectively done.
*
* Both collective communication and ncclCommInitRank can be used in conjunction
* of ncclGroupStart/ncclGroupEnd, but not together.
*
* Group semantics also allow to fuse multiple operations on the same device
* to improve performance (for aggregated collective calls), or to permit
* concurrent progress of multiple send/receive operations.
*/

/*
* Group Start
*
* Start a group call. All calls to NCCL until ncclGroupEnd will be fused into
* a single NCCL operation. Nothing will be started on the CUDA stream until
* ncclGroupEnd.
*/
ncclResult_t  ncclGroupStart();

/*
* Group End
*
* End a group call. Start a fused NCCL operation consisting of all calls since
* ncclGroupStart. Operations on the CUDA stream depending on the NCCL operations
* need to be called after ncclGroupEnd.
*/
ncclResult_t  ncclGroupEnd();

/*
* Group Simulate End
*
* Simulate a ncclGroupEnd() call and return NCCL's simulation info in a struct.
*/
ncclResult_t  ncclGroupSimulateEnd(ncclSimInfo_t* simInfo);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard