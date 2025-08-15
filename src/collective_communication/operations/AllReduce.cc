#include "custom_nccl.h"
#include <exception>
#include <stdio.h>
#include "common.h"
#include "cuda_runtime.h"
#include <string>
#include "mpi.h"

custom_ncclResult_t custom_AllReduce(){
    custom_ncclResult_t ret = custom_ncclSuccess;
    return ret;    
}
