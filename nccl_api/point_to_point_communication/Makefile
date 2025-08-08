BUILDDIR ?= ../../build
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY: build clean

# CUDA files
CU_FILES := $(wildcard *.cu)
TARGETS := $(CU_FILES:.cu=)

# Build targets
build: $(TARGETS)

# Compile each .cu file to executable
$(TARGETS): %: %.cu
	@echo "Building $@ from $<"
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $<
	@echo "Built $@ successfully"

clean:
	@echo "Cleaning p2p_communication..."
	rm -f $(TARGETS)
	@echo "Cleaned $(TARGETS)"
