BUILDDIR ?= ../../build
override BUILDDIR := $(abspath $(BUILDDIR))

$(info Received CXXFLAGS: $(CXXFLAGS))
$(info Received NVCCFLAGS: $(NVCCFLAGS))

.PHONY: build clean

TARGETS=algorithms operations

build: ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

%.clean:
	@echo "Cleaning collective_communication (does nothing for now)." $@