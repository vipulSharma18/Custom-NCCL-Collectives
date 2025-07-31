# Custom-NCCL-Collectives
Creating NCCL collectives from communication primitives in NCCL. This will create the API layer of NCCL while reusing the Transport layer and other NCCL code.

This is meant for educational purposes and to be used to demonstrate the multiple layers of abstraction involved in PyTorch's distributed training.

## References:
[1] S. Rennich, “CUDA C/C++ Streams and Concurrency”.
[2] Z. Hu et al., “Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms,” July 07, 2025, arXiv: arXiv:2507.04786. doi: 10.48550/arXiv.2507.04786.
[3] M. Keiblinger, M. Sieg, J. M. Ong, S. Jaghouar, and J. Hagemann, “Prime Collective Communications Library -- Technical Report,” May 20, 2025, arXiv: arXiv:2505.14065. doi: 10.48550/arXiv.2505.14065.
[4] “Quentin-Anthony/nanoMPI: Simple MPI implementation for prototyping or learning.” Accessed: July 22, 2025. [Online]. Available: https://github.com/Quentin-Anthony/nanoMPI
