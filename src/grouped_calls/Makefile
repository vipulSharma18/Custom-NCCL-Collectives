BUILDDIR ?= ../../build
GROUPED_BUILDDIR ?= $(BUILDDIR)/grouped_calls
override GROUPED_BUILDDIR := $(abspath $(GROUPED_BUILDDIR))

.PHONY: build clean

# CPP files might use CUDA runtime but don't define a CUDA function
CC_FILES := $(wildcard *.cc)
CU_FILES := $(wildcard *.cu)

CC_TARGETS := $(CC_FILES:.cc=.o)
CU_TARGETS := $(CU_FILES:.cu=)

# Build targets
build: $(CC_TARGETS:%.o=${GROUPED_BUILDDIR}/%.o) $(CU_TARGETS:%=${GROUPED_BUILDDIR}/%)

# Compile each .cu file to executable
${GROUPED_BUILDDIR}/%: %.cu
	@echo "Building $@ from $<"
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $< $(NVLDFLAGS)
	@echo "Built $@ successfully"

# compile only, not generating final executable with linking
${GROUPED_BUILDDIR}/%.o: %.cc
	@echo "Building $@ from $<"
	$(CXX) $(CXXFLAGS) -o $@ -c $<
	@echo "Built $@ successfully"

clean:
	@echo "Cleaning grouped_calls..."
	rm -rf $(GROUPED_BUILDDIR)
	@echo "Cleaned $(GROUPED_BUILDDIR)"
