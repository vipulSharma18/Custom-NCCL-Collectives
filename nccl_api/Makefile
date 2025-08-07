BUILDDIR ?= ../build
override BUILDDIR := $(abspath $(BUILDDIR))

CXXFLAGS += -I-I$(CURDIR)
NVCCFLAGS += -I-I$(CURDIR)

.PHONY: build clean

TARGETS=point_to_point_communication collective_communication

build:   ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR} CXXFLAGS="$(CXXFLAGS)" NVCCFLAGS="$(NVCCFLAGS)"

%.clean:
	${MAKE} -C $* clean BUILDDIR=${BUILDDIR}
