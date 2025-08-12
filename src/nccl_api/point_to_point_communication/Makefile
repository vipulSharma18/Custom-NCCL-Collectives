BUILDDIR ?= ../../build
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY: build clean

# CUDA files
CU_FILES := $(wildcard *.cu)
TARGETS := $(CU_FILES:.cu=)
CUSTOM_SUFFIX := custom_nccl

# Build targets
build: $(TARGETS:%=${BUILDDIR}/%_${CUSTOM_SUFFIX})

# Compile each .cu file to executable
${BUILDDIR}/%_${CUSTOM_SUFFIX}: %.cu
	@echo "Building $@ from $<"
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $< $(NVLDFLAGS)
	@echo "Built $@ successfully"

clean:
	@echo "Cleaning p2p_communication..."
	rm -f $(TARGETS:%=${BUILDDIR}/%_${CUSTOM_SUFFIX})
	@echo "Cleaned $(TARGETS:%=${BUILDDIR}/%_${CUSTOM_SUFFIX})"
