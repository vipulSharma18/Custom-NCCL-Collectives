BUILDDIR ?= ../../build
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY: build clean

TARGETS=algorithms operations

build: ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

%.clean:
	@echo "Cleaning collective_communication (does nothing for now)." $@