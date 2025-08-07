#include "nccl.h"
#include <stdio.h>
#include "common.h"


int sendRecv(ncclUniqueId Id){
    ncclGroupStart();

    ncclGroupEnd();
}


int main(){
    // init communicator
    ncclUniqueId Id;
    ncclGetUniqueId(&Id);
    int nranks = 3;
    ncclGroupStart();
    for(int rank=0; rank<nranks; rank++){
        //cudaSetDevice();
        //ncclCommInitRank(comm, nranks, Id,  rank);
    }
    ncclGroupEnd();

    printf("Running sendrecv.");
    char hostname[1024];                            \
    getHostName(hostname, 1024);
    sendRecv(Id);

    return 0;
}
