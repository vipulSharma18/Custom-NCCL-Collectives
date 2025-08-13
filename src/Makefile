BUILDDIR ?= ../build
override BUILDDIR := $(abspath $(BUILDDIR))

.EXPORT_ALL_VARIABLES:
CXXFLAGS += -I$(CURDIR)
NVCCFLAGS += -I$(CURDIR)

TARGETS=point_to_point_communication collective_communication 
P2P = $(BUILDDIR)/point_to_point_communication
COLLECTIVE = $(BUILDDIR)/collective_communication

.PHONY: build clean

build: ${P2P} ${COLLECTIVE} ${TARGETS:%=%.build} $(EXPORTED_LIB_TARGET)

${P2P}:
	mkdir -p ${P2P}

${COLLECTIVE}:
	mkdir -p ${COLLECTIVE}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

$(EXPORTED_LIB_TARGET): ${TARGETS:%=%.build}
	@echo "Exporting shared library $@ from $^"
	$(CXX) -shared -o $@ $(wildcard $(P2P)/*.o) $(wildcard $(COLLECTIVE)/*.o) $(LDFLAGS)
	@echo "Created shared library in build dir:"
	@ls -l $(BUILDDIR)

clean:
	@echo "Cleaning CUSTOM_NCCL build."
	rm -rf $(BUILDDIR)
	@echo "Cleaned CUSTOM_NCCL build $(BUILDDIR)"
