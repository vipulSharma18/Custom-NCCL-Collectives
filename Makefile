BUILDDIR ?= build
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY: all clean

TARGETS=nccl_api nccl_tests

all: ${BUILDDIR} ${TARGETS:%=%.build}
clean:
	rm -rf ${BUILDDIR}

${BUILDDIR}:
	mkdir -p ${BUILDDIR}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}
