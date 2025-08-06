# Custom-NCCL-Collectives
Creating NCCL collectives from communication primitives (send and recv). This will create most of the API layer of NCCL while reusing the Transport layer. 

The code is meant for educational purposes and to be used to demonstrate the multiple layers of abstraction involved in PyTorch's distributed training.

The project follows [2] in structuring the code, and the design of NCCL.

## Benchmarking:
* We use https://github.com/NVIDIA/nccl-tests/tree/master to test our collectives for correctness. Since the code is primarily to simplify and understand NCCL, and not optimized like NCCL, there is very little expectation of performance matching NCCL.

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

## References:
[1] S. Rennich, “CUDA C/C++ Streams and Concurrency”.    
[2] Z. Hu et al., “Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms,” July 07, 2025, arXiv: arXiv:2507.04786. doi: 10.48550/arXiv.2507.04786.    
[3] M. Keiblinger, M. Sieg, J. M. Ong, S. Jaghouar, and J. Hagemann, “Prime Collective Communications Library -- Technical Report,” May 20, 2025, arXiv: arXiv:2505.14065. doi: 10.48550/arXiv.2505.14065.    
[4] “Quentin-Anthony/nanoMPI: Simple MPI implementation for prototyping or learning.” Accessed: July 22, 2025. [Online]. Available: https://github.com/Quentin-Anthony/nanoMPI    
[5] “JSC Advanced Course: Using NCCL and NVSHMEM.” Accessed: July 06, 2025. [Online]. Available: https://juser.fz-juelich.de/record/1019178/files/02-NCCL_NVSHMEM.pdf      
