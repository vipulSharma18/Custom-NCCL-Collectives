BUILDDIR ?= ../../build
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY: build clean

CU_FILES := $(wildcard *.cu)
build: $(CU_FILES)
	@echo "Building collective_communication (does nothing for now). Files" $^

clean:
	@echo "Cleaning collective_communication (does nothing for now)."
