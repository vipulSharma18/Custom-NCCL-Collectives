#include "nccl.h"
#include "custom_nccl.h"
#include <cstring>

custom_ncclResult_t custom_ncclGetUniqueId(custom_ncclUniqueId* uniqueId){
    ncclUniqueId nccl_uniqueId;
    custom_ncclResult_t ret = custom_ncclResult_t(int(ncclGetUniqueId(&nccl_uniqueId)));
    memcpy(uniqueId->internal, nccl_uniqueId.internal, CUSTOM_NCCL_UNIQUE_ID_BYTES);
    return ret;
}

custom_ncclResult_t custom_ncclCommInitRank(
    custom_ncclComm_t* comm,
    int nranks,
    custom_ncclUniqueId commId,
    int rank
){
    ncclComm_t *nccl_comm = comm;
    ncclUniqueId nccl_commId;
    memcpy(nccl_commId.internal, commId.internal, CUSTOM_NCCL_UNIQUE_ID_BYTES);
    return ncclCommInitRank(nccl_comm, nranks, nccl_commId, rank);
}

custom_ncclResult_t custom_ncclCommDestroy(custom_ncclComm_t comm){
    ncclComm_t nccl_comm = comm;
    return ncclCommDestroy(nccl_comm);
}

const char* custom_ncclGetErrorString(custom_ncclResult_t result){
    ncclResult_t nccl_result = ncclResult_t(int(result));
    return ncclGetErrorString(nccl_result);
}

const char* custom_ncclGetLastError(custom_ncclComm_t comm){
    ncclComm_t nccl_comm = comm;
    return ncclGetLastError(nccl_comm);
}
