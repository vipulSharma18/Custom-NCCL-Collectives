#include "nccl.h"
#include "custom_nccl.h"


custom_ncclResult_t custom_ncclGroupStart(){
    return ncclGroupStart();
}

custom_ncclResult_t custom_ncclGroupEnd(){
    return ncclGroupEnd();
}
