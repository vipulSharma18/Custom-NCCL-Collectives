# NCCL From First Principles
Creating NCCL communication collectives and transport layer based on the description in [2]. The development is in phases, first making the communication API layer of NCCL while reusing the transport layer, and then creating the transport layer.

The code is for educational purposes and to demonstrate the multiple layers of abstraction involved in PyTorch's distributed training.

The project follows [2] in structuring the code, and the design of NCCL.

## Benchmarking:
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

## Build local NVIDIA-NCCL:
Building the main branch of NVIDIA's NCCL: https://github.com/NVIDIA/nccl/tree/master
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt install libnccl2 libnccl-dev
```

## Docker:
Git setup if needed:
```
(type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y
gh auth login
git config --global user.email "vipuls181999@gmail.com"
git config --global user.name "Vipul Sharma"
git clone git@github.com:vipulSharma18/Custom-NCCL-Collectives.git
```

Build and push to DockerHub:
```
docker build -t custom_nccl .
docker login -u dockervipul181999
docker tag custom_nccl:latest dockervipul181999/custom_nccl:latest
docker push dockervipul181999/custom_nccl:latest
```

Pull and run container:
```
docker pull dockervipul181999/custom_nccl:latest
docker run --gpus all -dit dockervipul181999/custom_nccl:latest
# for github setup
gh auth login
git config --global user.email "vipuls181999@gmail.com"
git config --global user.name "Vipul Sharma"
```

## Roadmap:

**Phase 1: NCCL APIs**
- [x] Setup NCCL_Tests for benchmarking and verifying correctness.
- [x] Initial DevOps setup
	- [x] Use a single makefile for the ported nccl_test repo and my own repo.
	- [x] Docker containerization to deploy on VastAI.
	- [x] GitHub workflow with Docker build and push to DockerHub setup to avoid slow local builds.
- [ ] Point2Point Communication APIs (NCCL grouped calls)
	- [ ] SendRecv
	- [ ] All-to-All
 	- [ ] All-to-One (Gather)
 	- [ ] Neighbor Exchange
 	- [ ] One-to-All (Scatter)
 	- [ ] RecvCopySend
 	- [ ] RecvReduceCopySend
 	- [ ] RecvReduceSend
 	- [ ] SendRecv
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

**Phase 2: Diving Deeper into NCCL, Transport Layer**
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


**Phase 3: Asynchronous GPU Communication, fault tolerance, and dynamic work group management**
- [ ] Explore Prime's Communication Collective Library (PCCL).

**Phase 4: NCCL Torch Integration**
- [ ] Process Group/Custom Distributed Backend for PyTorch.

**Phase 5: Simulating on-the-job debugging experience**
- [ ] Breaking NCCLs: Exploration of different bugs (possibly another repo with this repo as a submodule).

## References:
[1] S. Rennich, “CUDA C/C++ Streams and Concurrency”.    
[2] Z. Hu et al., “Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms,” July 07, 2025, arXiv: arXiv:2507.04786. doi: 10.48550/arXiv.2507.04786.    
[3] M. Keiblinger, M. Sieg, J. M. Ong, S. Jaghouar, and J. Hagemann, “Prime Collective Communications Library -- Technical Report,” May 20, 2025, arXiv: arXiv:2505.14065. doi: 10.48550/arXiv.2505.14065.    
[4] “Quentin-Anthony/nanoMPI: Simple MPI implementation for prototyping or learning.” Accessed: July 22, 2025. [Online]. Available: https://github.com/Quentin-Anthony/nanoMPI    
[5] “JSC Advanced Course: Using NCCL and NVSHMEM.” Accessed: July 06, 2025. [Online]. Available: https://juser.fz-juelich.de/record/1019178/files/02-NCCL_NVSHMEM.pdf      
