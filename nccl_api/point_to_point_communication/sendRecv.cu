#include "nccl.h"
#include <stdio.h>
#include "common.h"

int sendRecv(ncclUniqueId Id){
    ncclGroupStart();
    // TODO: Add actual send/recv operations here
    ncclGroupEnd();
    return 0;
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

    printf("Running sendrecv.\n");
    char hostname[1024];
    getHostName(hostname, 1024);
    printf("Hostname: %s\n", hostname);
    sendRecv(Id);

    return 0;
}
