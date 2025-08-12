#include "nccl.h"
#include <exception>
#include <stdio.h>
#include "common.h"
#include "cuda_runtime.h"
#include <string>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif

int recvSend(
    void* sendbuff,
    void* recvbuff,
    size_t size,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream
    ){
    // p2p calls within a group are independent, so the send and recv is done concurrently.
    ncclGroupStart();
    NCCLCHECK(ncclRecv(recvbuff, size, datatype, peer, comm, stream));
    NCCLCHECK(ncclSend(sendbuff, size, datatype, peer, comm, stream));
    ncclGroupEnd();
    return 0;
}

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
    float *sendbuff, *recvbuff;
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
    CUDACHECK(cudaMalloc(&sendbuff, size*sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size*sizeof(float)));

    // Create CUDA stream
    CUDACHECK(cudaStreamCreate(&stream));

    //init NCCL
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    printf("Running sendrecv.\n");
    if(myRank==0){
        recvSend((void*)sendbuff, (void*)recvbuff, size, ncclFloat32, 1, comm, stream);
    } else {
        recvSend((void*)sendbuff, (void*)recvbuff, size, ncclFloat32, 0, comm, stream);
    }

    //completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(stream));

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
