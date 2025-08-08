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

export MAKEFLAGS

.PHONY: build clean

TARGETS=point_to_point_communication collective_communication

build:   ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

%.clean:
	${MAKE} -C $* clean BUILDDIR=${BUILDDIR}
