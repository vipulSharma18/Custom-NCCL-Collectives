BUILDDIR ?= ../build
override BUILDDIR := $(abspath $(BUILDDIR))
TEST_BUILDDIR ?= $(BUILDDIR)/tests
override TEST_BUILDDIR := $(abspath $(TEST_BUILDDIR))

.EXPORT_ALL_VARIABLES:
NVLDFLAGS += -L$(BUILDDIR) -lcustom_nccl
LDFLAGS += -L$(BUILDDIR) -lcustom_nccl

.PHONY: build clean

TARGETS=unit_tests #nccl_perf_tests

build: ${TEST_BUILDDIR} ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

${TEST_BUILDDIR}:
	mkdir -p ${TEST_BUILDDIR}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

%.clean:
	${MAKE} -C $* clean BUILDDIR=${BUILDDIR}
