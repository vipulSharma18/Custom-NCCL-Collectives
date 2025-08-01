BUILDDIR ?= build
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY: all clean

all: nccl_api nccl_tests
	@echo "Make target 'all' - doing nothing for now"

nccl_api:
	cd nccl_api && make

nccl_tests:
	cd nccl_tests && make

clean:
	cd nccl_api && make clean
	cd nccl_tests && make clean