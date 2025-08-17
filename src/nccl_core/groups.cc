#include "nccl.h"
#include "custom_nccl.h"


custom_ncclResult_t custom_ncclGroupStart(){
    return result_converter(ncclGroupStart());
}

custom_ncclResult_t custom_ncclGroupEnd(){
    return result_converter(ncclGroupEnd());
}
