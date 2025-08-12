.EXPORT_ALL_VARIABLES:

BUILDDIR ?= ../build
override BUILDDIR := $(abspath $(BUILDDIR))

CXXFLAGS += -I$(CURDIR)
NVCCFLAGS += -I$(CURDIR)

CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDARTLIB ?= cudart
NCCL_HOME ?= /usr/lib/x86_64-linux-gnu
NCCLLIB ?= nccl

NVLDFLAGS  := -L${CUDA_LIB} -l${CUDARTLIB} -L${NCCL_HOME} -l${NCCLLIB}

# MPI headers and libs
NVCCFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi

.PHONY: build clean

TARGETS=point_to_point_communication collective_communication

build:   ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

%.clean:
	${MAKE} -C $* clean BUILDDIR=${BUILDDIR}
