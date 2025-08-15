## Disclaimer:

This folder is the core part of nccl that we've reused to bootstrap our custom NCCL.
As the project makes progress, more and more code will go from being bootstrapped and a wrapper
around NCCL APIs, to being implemented from scratch.

This directory should be the only directory where nccl.h is included. All the rest of the code is free from nccl.h.