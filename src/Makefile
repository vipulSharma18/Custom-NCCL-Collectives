BUILDDIR ?= ../build
override BUILDDIR := $(abspath $(BUILDDIR))

.EXPORT_ALL_VARIABLES:
CXXFLAGS += -fPIC

TARGETS=nccl_core grouped_calls collective_communication
NCCL_CORE = $(BUILDDIR)/nccl_core
COLLECTIVE = $(BUILDDIR)/collective_communication
GROUPED = $(BUILDDIR)/grouped_calls

.PHONY: build clean

build: ${NCCL_CORE} ${GROUPED} ${COLLECTIVE} ${TARGETS:%=%.build} $(EXPORTED_LIB_TARGET)

${COLLECTIVE}:
	mkdir -p ${COLLECTIVE}

${GROUPED}:
	mkdir -p ${GROUPED}

${NCCL_CORE}:
	mkdir -p ${NCCL_CORE}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

$(EXPORTED_LIB_TARGET): ${TARGETS:%=%.build}
	@echo "Exporting shared library $@ from $^"
	$(CXX) -shared -o $@ \
		$(wildcard $(NCCL_CORE)/*.o) \
		$(wildcard $(COLLECTIVE)/*.o) \
		$(wildcard $(GROUPED)/*.o) \
		$(LDFLAGS)
	@echo "Created shared library in build dir:"
	@ls -l $(BUILDDIR)

clean:
	@echo "Cleaning CUSTOM_NCCL build."
	rm -rf $(BUILDDIR)
	@echo "Cleaned CUSTOM_NCCL build $(BUILDDIR)"
