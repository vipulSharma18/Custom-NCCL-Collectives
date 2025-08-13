/*************************************************************************
* Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
*
* See LICENSE.txt for license information
************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#include "nccl.h"
#include "custom_nccl.h"
#include <cstring>
#include <stdio.h>
#include <cstdint>
#include <algorithm>
#include <string>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>

// For nccl.h < 2.13 since we define a weak fallback
extern "C" char const* ncclGetLastError(ncclComm_t comm);

#define CUDACHECK(cmd) do {                         \
    cudaError_t err = cmd;                            \
    if( err != cudaSuccess ) {                        \
        char hostname[1024];                            \
        getHostName(hostname, 1024);                    \
        printf("%s: Test CUDA failure %s:%d '%s'\n",    \
            hostname,                                  \
            __FILE__,__LINE__,cudaGetErrorString(err)); \
        return 1;                           \
    }                                                 \
} while(0)

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,13,0)
#define NCCLCHECK(cmd) do {                         \
    ncclResult_t res = cmd;                           \
    if (res != ncclSuccess) {                         \
        char hostname[1024];                            \
        getHostName(hostname, 1024);                    \
        printf("%s: Test NCCL failure %s:%d "           \
            "'%s / %s'\n",                           \
            hostname,__FILE__,__LINE__,              \
            ncclGetErrorString(res),                 \
            ncclGetLastError(NULL));                 \
        return 1;                           \
    }                                                 \
} while(0)
#else
#define NCCLCHECK(cmd) do {                         \
    ncclResult_t res = cmd;                           \
    if (res != ncclSuccess) {                         \
        char hostname[1024];                            \
        getHostName(hostname, 1024);                    \
        printf("%s: Test NCCL failure %s:%d '%s'\n",    \
            hostname,                                  \
            __FILE__,__LINE__,ncclGetErrorString(res)); \
        return testNcclError;                           \
    }                                                 \
} while(0)
#endif

#ifdef MPI_SUPPORT
#define MPICHECK(cmd) do {                          \
    int e = cmd;                                      \
    if( e != MPI_SUCCESS ) {                          \
      printf("Failed: MPI error %s:%d '%d'\n",        \
          __FILE__,__LINE__, e);   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)  
#endif

#define CUSTOMNCCLCHECK(cmd) do {                         \
    custom_ncclResult_t res = cmd;                           \
    if (res != custom_ncclSuccess) {                         \
        char hostname[1024];                            \
        getHostName(hostname, 1024);                    \
        printf("%s: Test NCCL failure %s:%d "           \
            "'%s / %s'\n",                           \
            hostname,__FILE__,__LINE__,              \
            ncclGetErrorString(res),                 \
            ncclGetLastError(NULL));                 \
        return 1;                           \
    }                                                 \
} while(0)

static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

static uint64_t getHash(const char* string, size_t n) {
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (size_t c = 0; c < n; c++) {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

/* Generate a hash of the unique identifying string for this host
* that will be unique for both bare-metal and container instances
* Equivalent of a hash of;
*
* $(hostname)$(cat /proc/sys/kernel/random/boot_id)
*
*/
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t getHostHash(const char* hostname) {
    char hostHash[1024];

    // Fall back is the hostname if something fails
    (void) strncpy(hostHash, hostname, sizeof(hostHash));
    int offset = strlen(hostHash);

    FILE *file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
        char *p;
        if (fscanf(file, "%ms", &p) == 1) {
            strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
            free(p);
        }
    }
    fclose(file);

    // Make sure the string is terminated
    hostHash[sizeof(hostHash)-1]='\0';

    return getHash(hostHash, strlen(hostHash));
}

extern int is_main_proc;
extern thread_local int is_main_thread;
#define PRINT if (is_main_thread) printf

#endif
