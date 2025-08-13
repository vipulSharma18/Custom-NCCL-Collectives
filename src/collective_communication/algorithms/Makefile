BUILDDIR ?= ../../build
COLLECTIVE_BUILDDIR ?= $(BUILDDIR)/collective_communication
override COLLECTIVE_BUILDDIR := $(abspath $(COLLECTIVE_BUILDDIR))
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY: build clean

CC_FILES := $(wildcard *.cc)
CU_FILES := $(wildcard *.cu)

CC_TARGETS := $(CC_FILES:.cc=.o)
CU_TARGETS := $(CU_FILES:.cu=)

build: $(CC_TARGETS:%.o=${COLLECTIVE_BUILDDIR}/%.o) $(CU_TARGETS:%=${COLLECTIVE_BUILDDIR}/%)

# Compile each .cu file to executable
${COLLECTIVE_BUILDDIR}/%: %.cu
	@echo "Building $@ from $<"
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $< $(NVLDFLAGS)
	@echo "Built $@ successfully"

# compile only, not generating final executable with linking
${COLLECTIVE_BUILDDIR}/%.o: %.cc
	@echo "Building $@ from $<"
	$(CXX) $(CXXFLAGS) -o $@ -c $<
	@echo "Built $@ successfully"

clean:
	@echo "Cleaning collective_communication..."
	rm -f $(CC_TARGETS:%.o=${COLLECTIVE_BUILDDIR}/%.o) $(CU_TARGETS:%=${COLLECTIVE_BUILDDIR}/%)
	@echo "Cleaned $(CC_TARGETS:%.o=${COLLECTIVE_BUILDDIR}/%.o) $(CU_TARGETS:%=${COLLECTIVE_BUILDDIR}/%)"
