BUILDDIR ?= ../build
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY: build clean

TARGETS=nccl_perf_tests unit_tests

build: ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

%.clean:
	${MAKE} -C $* clean BUILDDIR=${BUILDDIR}
