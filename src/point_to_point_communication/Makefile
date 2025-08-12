BUILDDIR ?= ../../build
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY: build clean

# CPP files might use CUDA runtime but don't define a CUDA function
CC_FILES := $(wildcard *.cc)
CU_FILES := $(wildcard *.cu)

CC_TARGETS := $(CC_FILES:.cc=)
CU_TARGETS := $(CU_FILES:.cu=)

CUSTOM_SUFFIX := custom_nccl

# Build targets
build: $(CC_TARGETS:%=${BUILDDIR}/%_${CUSTOM_SUFFIX}) $(CU_TARGETS:%=${BUILDDIR}/%_${CUSTOM_SUFFIX})

# Compile each .cu file to executable
${BUILDDIR}/%_${CUSTOM_SUFFIX}: %.cu
	@echo "Building $@ from $<"
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $< $(NVLDFLAGS)
	@echo "Built $@ successfully"

# compile only, not generating final executable with linking
${BUILDDIR}/%_${CUSTOM_SUFFIX}: %.cc
	@echo "Building $@ from $<"
	$(CXX) $(CXXFLAGS) -o $@ -c $<
	@echo "Built $@ successfully"

clean:
	@echo "Cleaning p2p_communication..."
	rm -f $(CC_TARGETS:%=${BUILDDIR}/%_${CUSTOM_SUFFIX}) $(CU_TARGETS:%=${BUILDDIR}/%_${CUSTOM_SUFFIX})
	@echo "Cleaned $(CC_TARGETS:%=${BUILDDIR}/%_${CUSTOM_SUFFIX}) $(CU_TARGETS:%=${BUILDDIR}/%_${CUSTOM_SUFFIX})"
