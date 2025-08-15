BUILDDIR ?= ../../build
NCCL_CORE_BUILDDIR ?= $(BUILDDIR)/nccl_core
override NCCL_CORE_BUILDDIR := $(abspath $(NCCL_CORE_BUILDDIR))

.PHONY: build clean

# CPP files might use CUDA runtime but don't define a CUDA function
CC_FILES := $(wildcard *.cc)
CU_FILES := $(wildcard *.cu)

CC_TARGETS := $(CC_FILES:.cc=.o)
CU_TARGETS := $(CU_FILES:.cu=)

# Build targets
build: $(CC_TARGETS:%.o=${NCCL_CORE_BUILDDIR}/%.o) $(CU_TARGETS:%=${NCCL_CORE_BUILDDIR}/%)

# Compile each .cu file to executable
${NCCL_CORE_BUILDDIR}/%: %.cu
	@echo "Building $@ from $<"
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $< $(NVLDFLAGS)
	@echo "Built $@ successfully"

# compile only, not generating final executable with linking
${NCCL_CORE_BUILDDIR}/%.o: %.cc
	@echo "Building $@ from $<"
	$(CXX) $(CXXFLAGS) -o $@ -c $<
	@echo "Built $@ successfully"

clean:
	@echo "Cleaning NCCL_CORE BUILDDIR..."
	rm -rf $(NCCL_CORE_BUILDDIR)
	@echo "Cleaned $(NCCL_CORE_BUILDDIR)"
