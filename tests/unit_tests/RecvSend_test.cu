#include "custom_nccl.h"
#include "nccl.h"
#include "common.h"
#include "cuda_runtime.h"
#include <cassert>
#include <stdio.h>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif

int main(int argc, char* argv[]){

    int myRank, nRanks = 0;

    //doing multiprocessing with MPI. init MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    assert(nRanks == 2);

    // main process's creates a shared single nccl communicator ID
    ncclUniqueId id;
    ncclComm_t comm;
    int *sendbuff, *recvbuff, *userbuff;
    cudaStream_t stream;
    int size = 1;

    // create unique id for communicator on rank 0 and bcast it to other ranks using MPI
    if(myRank == 0){
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // set proc to gpu
    CUDACHECK(cudaSetDevice(myRank));

    // Allocate device buffers
    CUDACHECK(cudaMalloc(&sendbuff, size*sizeof(int)));
    CUDACHECK(cudaMalloc(&recvbuff, size*sizeof(int)));

    // Create CUDA stream
    CUDACHECK(cudaStreamCreate(&stream));

    //init NCCL
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    userbuff = &myRank;
    sendbuff = userbuff;
    printf("Data in sendbuff of rank %d is %d.\n", myRank, *sendbuff); 
    printf("Running sendrecv.\n");
    if(myRank==0){
        NCCLCHECK(
            custom_RecvSend((const void*)sendbuff, (void*)recvbuff, size, ncclInt8, 1, comm, stream)
        );
    } else {
        NCCLCHECK(
            custom_RecvSend((const void*)sendbuff, (void*)recvbuff, size, ncclInt8, 0, comm, stream)
        );
    }

    //completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(stream));

    printf("CUDA Stream sync-ed.\n");
    userbuff = recvbuff;
    printf("Data in recvbuff of rank %d is %d.\n", myRank, *recvbuff); 
    assert(*recvbuff == (nRanks-myRank)/nRanks);

    //free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));

    // Destroy CUDA stream
    CUDACHECK(cudaStreamDestroy(stream));
    
    //finalize NCCL
    NCCLCHECK(ncclCommDestroy(comm));

    //finalizing MPI
    MPICHECK(MPI_Finalize());

    printf("[MPI Rank %d] Success \n", myRank);
    return 0;
}
