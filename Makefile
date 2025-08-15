.EXPORT_ALL_VARIABLES:
BUILDDIR ?= build
override BUILDDIR := $(abspath $(BUILDDIR))

CUSTOM_NCCL_INCLUDE ?= $(CURDIR)/src/include
EXPORTED_LIB_NAME = libcustom_nccl.so
EXPORTED_LIB_TARGET = $(BUILDDIR)/$(EXPORTED_LIB_NAME)

CUDA_HOME ?= /usr/local/cuda
CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
CUDARTLIB ?= cudart

NVCC ?= $(CUDA_HOME)/bin/nvcc
NCCL_HOME ?= /usr/lib/x86_64-linux-gnu
NCCLLIB ?= nccl

MPI_HOME ?= /usr/lib/x86_64-linux-gnu/openmpi

NVCCFLAGS := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11 -O3 -g -DMPI_SUPPORT -I$(MPI_HOME)/include -I$(CUSTOM_NCCL_INCLUDE)
CXXFLAGS := -std=c++11 -O3 -g -I$(CUDA_INC) -DMPI_SUPPORT -I$(MPI_HOME)/include -I$(CUSTOM_NCCL_INCLUDE)

NVLDFLAGS := -L${CUDA_LIB} -l${CUDARTLIB} -L${NCCL_HOME} -l${NCCLLIB} -L$(MPI_HOME)/lib -lmpi -lmpi_cxx
LDFLAGS := -L${CUDA_LIB} -l${CUDARTLIB} -L${NCCL_HOME} -l${NCCLLIB} -L$(MPI_HOME)/lib -lmpi -lmpi_cxx

CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 13; echo $$?),0)
# Add Blackwell but drop Pascal & Volta support if we're using CUDA13.0 or above
NVCC_GENCODE ?= -gencode=arch=compute_75,code=sm_75 \
		-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_90,code=sm_90 \
		-gencode=arch=compute_100,code=sm_100 \
		-gencode=arch=compute_120,code=sm_120 \
		-gencode=arch=compute_120,code=compute_120
else ifeq ($(shell test "0$(CUDA_MAJOR)" -eq 12 -a "0$(CUDA_MINOR)" -ge 8; echo $$?),0)
# Include Blackwell support if we're using CUDA12.8 or above
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
		-gencode=arch=compute_61,code=sm_61 \
		-gencode=arch=compute_70,code=sm_70 \
		-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_90,code=sm_90 \
		-gencode=arch=compute_100,code=sm_100 \
		-gencode=arch=compute_120,code=sm_120 \
		-gencode=arch=compute_120,code=compute_120
else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 12; echo $$?),0)
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
		-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_90,code=sm_90 \
		-gencode=arch=compute_90,code=compute_90
else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
		-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_80,code=compute_80
else
NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_70,code=compute_70
endif

.PHONY: all clean

TARGETS=src tests

all: ${BUILDDIR} ${TARGETS:%=%.build}
clean:
	rm -rf ${BUILDDIR}

${BUILDDIR}:
	mkdir -p ${BUILDDIR}

src.build: ${BUILDDIR}
	${MAKE} -C src build BUILDDIR=${BUILDDIR}

tests.build: src.build ${BUILDDIR}
	${MAKE} -C tests build BUILDDIR=${BUILDDIR}
