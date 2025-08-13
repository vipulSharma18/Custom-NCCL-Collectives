BUILDDIR ?= ../../build
COLLECTIVE_BUILDDIR ?= $(BUILDDIR)/collective_communication
override COLLECTIVE_BUILDDIR := $(abspath $(COLLECTIVE_BUILDDIR))
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY: build clean

TARGETS=algorithms operations

build: ${TARGETS:%=%.build}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR} COLLECTIVE_BUILDDIR=${COLLECTIVE_BUILDDIR}

clean:
	@echo "Cleaning collective_communication build files."
	rm -rf $(COLLECTIVE_BUILDDIR)
	@echo "Cleaned $(COLLECTIVE_BUILDDIR)"