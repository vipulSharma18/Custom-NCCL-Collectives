#include "nccl.h"
#include "custom_nccl.h"


custom_ncclResult_t custom_ncclGetUniqueId(custom_ncclUniqueId* uniqueId){
    ncclUniqueId* nccl_uniqueId = &ncclUniqueId(*uniqueId);
    return ncclGetUniqueId(nccl_uniqueId);
}

custom_ncclResult_t custom_ncclCommInitRank(
    custom_ncclComm_t* comm,
    int nranks,
    custom_ncclUniqueId commId,
    int rank
){
    ncclComm_t* nccl_comm = &ncclComm_t(*comm);
    ncclUniqueId nccl_commId = ncclUniqueId(commId);
    return ncclCommInitRank(nccl_comm, nranks, nccl_commId, rank);
}

custom_ncclResult_t custom_ncclCommDestroy(custom_ncclComm_t comm){
    ncclComm_t nccl_comm = ncclComm_t(comm);
    return ncclCommDestroy(nccl_comm);
}

const char* custom_ncclGetErrorString(custom_ncclResult_t result){
    ncclResult_t nccl_result = ncclResult_t(int(result));
    return ncclGetErrorString(nccl_result);
}

const char* custom_ncclGetLastError(custom_ncclComm_t comm){
    ncclComm_t nccl_comm = ncclComm_t(comm);
    return ncclGetLastError(nccl_comm);
}

