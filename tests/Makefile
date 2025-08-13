BUILDDIR ?= ../build
override BUILDDIR := $(abspath $(BUILDDIR))

.EXPORT_ALL_VARIABLES
NVLDFLAGS += -L$(BUILDDIR) -llibcustom_nccl
LDFLAGS += -L$(BUILDDIR) -llibcustom_nccl

.PHONY: build clean

TARGETS=unit_tests #nccl_perf_tests

build: ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

%.clean:
	${MAKE} -C $* clean BUILDDIR=${BUILDDIR}
