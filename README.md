# Custom-NCCL-Collectives
Creating NCCL collectives from communication primitives (send and recv). This will create most of the API layer of NCCL while reusing the Transport layer. 

The code is meant for educational purposes and to be used to demonstrate the multiple layers of abstraction involved in PyTorch's distributed training.

The project follows [2] in structuring the code, and the design of NCCL.

## Benchmarking:
* We use https://github.com/NVIDIA/nccl-tests/tree/master to test our collectives for correctness. Since the code is primarily to simplify and understand NCCL, and not optimized like NCCL, there is very little expectation of performance matching NCCL.

## References:
[1] S. Rennich, “CUDA C/C++ Streams and Concurrency”.
[2] Z. Hu et al., “Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms,” July 07, 2025, arXiv: arXiv:2507.04786. doi: 10.48550/arXiv.2507.04786.
[3] M. Keiblinger, M. Sieg, J. M. Ong, S. Jaghouar, and J. Hagemann, “Prime Collective Communications Library -- Technical Report,” May 20, 2025, arXiv: arXiv:2505.14065. doi: 10.48550/arXiv.2505.14065.
[4] “Quentin-Anthony/nanoMPI: Simple MPI implementation for prototyping or learning.” Accessed: July 22, 2025. [Online]. Available: https://github.com/Quentin-Anthony/nanoMPI
[5] “JSC Advanced Course: Using NCCL and NVSHMEM.” Accessed: July 06, 2025. [Online]. Available: https://juser.fz-juelich.de/record/1019178/files/02-NCCL_NVSHMEM.pdf
