> Caution:

Need to add the shared object file to LD_LIBRARY_PATH or do the below:
```
.EXPORT_ALL_VARIABLES:
# -Wl,-rpath, embeds the library path in the executable, 
# so we don't need to modify LD_LIBRARY_PATH env var
NVLDFLAGS += -L$(BUILDDIR) -lcustom_nccl -Xlinker -rpath -Xlinker $(BUILDDIR)
LDFLAGS += -L$(BUILDDIR) -lcustom_nccl -Wl,-rpath,$(BUILDDIR)
```