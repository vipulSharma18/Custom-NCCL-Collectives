BUILDDIR ?= ../build
override BUILDDIR := $(abspath $(BUILDDIR))
TEST_BUILDDIR ?= $(BUILDDIR)/tests
override TEST_BUILDDIR := $(abspath $(TEST_BUILDDIR))

.EXPORT_ALL_VARIABLES:
# -Wl,-rpath, embeds the library path in the executable, 
# so we don't need to modify LD_LIBRARY_PATH env var
NVLDFLAGS += -L$(BUILDDIR) -lcustom_nccl -Xlinker -rpath -Xlinker $(BUILDDIR)
LDFLAGS += -L$(BUILDDIR) -lcustom_nccl -Wl,-rpath,$(BUILDDIR)

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
