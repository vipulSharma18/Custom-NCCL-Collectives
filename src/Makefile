BUILDDIR ?= ../build
override BUILDDIR := $(abspath $(BUILDDIR))

.EXPORT_ALL_VARIABLES:
CXXFLAGS += -I$(CURDIR) -fPIC
NVCCFLAGS += -I$(CURDIR)

TARGETS=point_to_point_communication collective_communication grouped_calls
P2P = $(BUILDDIR)/point_to_point_communication
COLLECTIVE = $(BUILDDIR)/collective_communication
GROUPED = $(BUILDDIR)/grouped_calls

.PHONY: build clean

build: ${P2P} ${COLLECTIVE} ${GROUPED} ${TARGETS:%=%.build} $(EXPORTED_LIB_TARGET)

${P2P}:
	mkdir -p ${P2P}

${COLLECTIVE}:
	mkdir -p ${COLLECTIVE}

${GROUPED}:
	mkdir -p ${GROUPED}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

$(EXPORTED_LIB_TARGET): ${TARGETS:%=%.build}
	@echo "Exporting shared library $@ from $^"
	$(CXX) -shared -o $@ \
		$(wildcard $(P2P)/*.o) \
		$(wildcard $(COLLECTIVE)/*.o) \
		$(wildcard $(GROUPED)/*.o) \
		$(LDFLAGS)
	@echo "Created shared library in build dir:"
	@ls -l $(BUILDDIR)

clean:
	@echo "Cleaning CUSTOM_NCCL build."
	rm -rf $(BUILDDIR)
	@echo "Cleaned CUSTOM_NCCL build $(BUILDDIR)"
