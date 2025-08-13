BUILDDIR ?= ../../build
TEST_BUILDDIR ?= $(BUILDDIR)/tests
override TEST_BUILDDIR := $(abspath $(TEST_BUILDDIR))

.PHONY: build clean

# CPP files might use CUDA runtime but don't define a CUDA function
CC_FILES := $(wildcard *.cc)
CU_FILES := $(wildcard *.cu)

CC_TARGETS := $(CC_FILES:.cc=.o)
CU_TARGETS := $(CU_FILES:.cu=)

# Build targets
build: $(CC_TARGETS:%.o=${P2P_BUILDDIR}/%.o) $(CU_TARGETS:%=${P2P_BUILDDIR}/%)

# Compile each .cu file to executable
${P2P_BUILDDIR}/%: %.cu
	@echo "Building $@ from $<"
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $< $(NVLDFLAGS)
	@echo "Built $@ successfully"

# compile only, not generating final executable with linking
${P2P_BUILDDIR}/%.o: %.cc
	@echo "Building $@ from $<"
	$(CXX) $(CXXFLAGS) -o $@ -c $<
	@echo "Built $@ successfully"

clean:
	@echo "Cleaning tests."
	rm -f $(TEST_BUILDDIR)
	@echo "Cleaned $(TEST_BUILDDIR)"
