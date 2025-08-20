# NCCL From First Principles

[![Docker Build](https://github.com/vipulSharma18/NCCL-From-First-Principles/actions/workflows/build_and_deploy.yml/badge.svg?branch=main)](https://github.com/vipulSharma18/NCCL-From-First-Principles/actions/workflows/build_and_deploy.yml) [![Run with VastAI](https://img.shields.io/badge/Run_on-Vast.ai-purple?logo=google-cloud&logoColor=white)](https://cloud.vast.ai?ref_id=288801&template_id=2e27cd968cd6da34006dba9cc06c897b)

Creating NCCL communication collectives and transport layer based on the description in [2]. The development is in phases, first making the communication API layer of NCCL while reusing the transport layer, and then creating the transport layer.

The code is for educational purposes and to demonstrate the multiple layers of abstraction involved in PyTorch's distributed training.

The project follows [2] in structuring the code, and the design of NCCL.

## Run with Docker:
Pull and run container:
```
docker pull ghcr.io/vipulsharma18/nccl-from-first-principles:main
docker run --gpus all -dit ghcr.io/vipulsharma18/nccl-from-first-principles:main
```

Git setup if needed within the container:
```
gh auth login
git config --global user.email "vipuls181999@gmail.com"
git config --global user.name "Vipul Sharma"
```

## Test Custom NCCL:

* Include custom_nccl.h from src/include, and link the libcustom_nccl.so object from build dir.
* To run tests, simply run executable in the build/tests directory like below:

```
mpirun -n 2 ./build/tests/SendRecv_test
```
> Note: MPI might require running as a non-root user, to create and run as a non-root user run this:

```
adduser test
sudo -u test mpirun -n 2 ./build/tests/SendRecv_test > out.log 2>&1
```

For debugging tests if 0x1509 is the line in the stack trace:
```
addr2line -e build/tests/SendRecv_test 0x1509
```

## Roadmap:

**Phase 1: NCCL APIs**
- [x] Setup NCCL_Tests for benchmarking and verifying correctness.
- [x] Initial DevOps setup
	- [x] Use a single makefile for the ported nccl_test repo and my own repo.
	- [x] Docker containerization to deploy on VastAI.
	- [x] GitHub workflow with Docker build and push to DockerHub setup to avoid slow local builds.
- [ ] Point2Point Communication APIs (NCCL grouped calls)
	- [x] SendRecv (1 Peer Exchange)
	- [ ] All-to-All
 	- [ ] All-to-One (Gather)
 	- [ ] Neighbor Exchange
 	- [ ] One-to-All (Scatter)
 	- [ ] RecvCopySend
 	- [ ] RecvReduceCopySend
 	- [ ] RecvReduceSend
- [ ] Collective Communication APIs (both Ring and Tree implementations using P2P APIs)
	- [ ] Ring
 		- [ ] AllGather
		- [ ] AllReduce
 		- [ ] Broadcast
 		- [ ] Reduce
 		- [ ] ReduceScatter
 	- [ ] Tree
 		- [ ] AllGather
		- [ ] AllReduce
 		- [ ] Broadcast
 		- [ ] Reduce
 		- [ ] ReduceScatter

**Phase 2: NCCL Torch Integration**
- [ ] Creating Process Group/Custom Distributed Backend for PyTorch.
- [ ] Integrating the project with the Abstraction Layers of GPU Parallelism project: https://github.com/vipulSharma18/The-Abstraction-Layers-of-GPU-Parallelism/tree/main

**Phase 3: Diving Deeper into NCCL, Transport Layer**
- [ ] Intranode Data Transfer
	- [ ] P2P (IPC/CUDA Memory): Data transfer between two GPUs via PCIe bus, or NVLink, via an intermediate FIFO buffer.
	- [ ] P2P (Direct): Support direct access to device buffers over NVLink and PCIe.
 	- [ ] Shared Memory: GPU mem to DMA engine to PCIe bus to host memory to PCIe bus to receiver DMA engine to received device buffer.

Figure 1 from [2]:

<img width="675" height="460" alt="image" src="https://github.com/user-attachments/assets/9a07f55c-4e6a-4eb5-8c22-c435adcf80bc" />

- [ ] Internode Data Transfer
	- [ ] Infiniband Verbs Transport: RDMA with minimal CPU involvement, except when GPU memory isn't free/accessible directly and it uses host memory. 
 	- [ ] Sockets: CPU-managed transport with intermediate buffers as pinned host memory.

Figure 2 from [2]:

<img width="648" height="642" alt="image" src="https://github.com/user-attachments/assets/8ae4ee1e-9f0d-4292-8e49-299287240e8d" />

**Phase 4: Asynchronous GPU Communication, fault tolerance, and dynamic work group management**
- [ ] Explore Prime's Communication Collective Library (PCCL).

**Phase 5: Simulating on-the-job debugging experience**
- [ ] Breaking NCCLs: Exploration of different bugs (possibly another repo with this repo as a submodule).

## Devtools:
* For setting up clangd, use bear (`apt-get install bear`) like:
```
make clean; bear -- make all
```
This will generate a compile_commands.json file that can be used by clangd server after you
restart it (Ctrl+Shift+P, clangd:restart language server).

## (Deprecated) Benchmarking with NCCL_PERF:
* We use https://github.com/NVIDIA/nccl-tests/tree/master to test our collectives for correctness. Since the code is primarily to simplify and understand NCCL, and not optimized like NCCL, there is very little expectation of performance matching NCCL.

NCCL tests repo supports multiple processes, multiple threads, and multiple CUDA devices per thread testing. It tests for both correctness and performance, but we're only interested in correctness.

When doing multi-process testing, NCCL tests uses MPI. Total ranks = num procs * num threads per proc * num GPUs per thread.

For running the tests on a single node:
```
$ make all
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

## References (You can find notes and annotated papers in the research_notes folder):
[1] S. Rennich, “CUDA C/C++ Streams and Concurrency”.    
[2] Z. Hu et al., “Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms,” July 07, 2025, arXiv: arXiv:2507.04786. doi: 10.48550/arXiv.2507.04786.    
[3] M. Keiblinger, M. Sieg, J. M. Ong, S. Jaghouar, and J. Hagemann, “Prime Collective Communications Library -- Technical Report,” May 20, 2025, arXiv: arXiv:2505.14065. doi: 10.48550/arXiv.2505.14065.    
[4] “Quentin-Anthony/nanoMPI: Simple MPI implementation for prototyping or learning.” Accessed: July 22, 2025. [Online]. Available: https://github.com/Quentin-Anthony/nanoMPI    
[5] “JSC Advanced Course: Using NCCL and NVSHMEM.” Accessed: July 06, 2025. [Online]. Available: https://juser.fz-juelich.de/record/1019178/files/02-NCCL_NVSHMEM.pdf      
