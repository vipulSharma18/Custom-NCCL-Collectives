NCCL tests repo supports multiple processes, multiple threads, and multiple CUDA devices per thread testing. It tests for both correctness and performance, but we're only interested in correctness.

When doing multi-process testing, NCCL tests uses MPI. Total ranks = num procs * num threads per proc * num GPUs per thread.

For running the tests on a single node:
```
$ make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
# test command for 1 gpu 1 thread, just to ensure things are working fine.
$ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
# single thread with 8 GPUs.
$ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

If we want to test on multiple processes/multiple nodes, we need to compile the tests with MPI support.
```
$ make MPI=1 NAME_SUFFIX=_mpi MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
$ mpirun -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

Flags:

**GPUs**: -t for num of threads per proc, -g for num gpus per thread.    
**Message size**: -b for min size of message in bytes, -e for max size in bytes, -f increment multiplication factor between 2 sizes.    
**Op args**: -o for reduction op. -d for datatype. -r for root rank.    

Reference: https://github.com/NVIDIA/nccl-tests/blob/master/README.md