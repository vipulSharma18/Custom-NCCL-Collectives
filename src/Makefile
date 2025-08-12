BUILDDIR ?= ../build
override BUILDDIR := $(abspath $(BUILDDIR))

.EXPORT_ALL_VARIABLES:
CXXFLAGS += -I$(CURDIR) 
NVCCFLAGS += -I$(CURDIR)

.PHONY: build clean

TARGETS=point_to_point_communication collective_communication

build: ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

%.clean:
	${MAKE} -C $* clean BUILDDIR=${BUILDDIR}
