#pragma once
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <hip/hip_runtime.h>

#define GPU_MAX_NBOR_SIZE 4096
#define hipErrcheck(res) {hipAssert((res), __FILE__, __LINE__);}
inline void hipAssert(hipError_t code, const char *file, int line, bool abort=true) {
  if (code != hipSuccess) {
    fprintf(stderr,"hip assert: %s %s %d\n", hipGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// #ifdef __HIPCC__
// static __inline__ __device__ double atomicAdd(
//     double* address, 
//     double val) 
// {
//   unsigned long long int* address_as_ull = (unsigned long long int*)address;
//   unsigned long long int old = *address_as_ull, assumed;
//   do {
//     assumed = old;
//     old = atomicCAS(address_as_ull, assumed,
//           __double_as_longlong(val + __longlong_as_double(assumed)));
//   // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
//   } while (assumed != old);
//   return __longlong_as_double(old);
// }
// #endif

namespace deepmd {
template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const std::vector<FPTYPE> &host) 
{
  hipErrcheck(hipMemcpy(device, &host[0], sizeof(FPTYPE) * host.size(), hipMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_host_to_device(
    FPTYPE * device, 
    const FPTYPE * host,
    const int size) 
{
  hipErrcheck(hipMemcpy(device, host, sizeof(FPTYPE) * size, hipMemcpyHostToDevice));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    std::vector<FPTYPE> &host) 
{
  hipErrcheck(hipMemcpy(&host[0], device, sizeof(FPTYPE) * host.size(), hipMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void memcpy_device_to_host(
    const FPTYPE * device, 
    FPTYPE * host,
    const int size) 
{
  hipErrcheck(hipMemcpy(host, device, sizeof(FPTYPE) * size, hipMemcpyDeviceToHost));  
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const std::vector<FPTYPE> &host) 
{
  hipErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
}

template <typename FPTYPE>
void malloc_device_memory(
    FPTYPE * &device, 
    const int size) 
{
  hipErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * size));
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const std::vector<FPTYPE> &host) 
{
  hipErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * host.size()));
  memcpy_host_to_device(device, host);
}

template <typename FPTYPE>
void malloc_device_memory_sync(
    FPTYPE * &device,
    const FPTYPE * host,
    const int size)
{
  hipErrcheck(hipMalloc((void **)&device, sizeof(FPTYPE) * size));
  memcpy_host_to_device(device, host, size);
}

template <typename FPTYPE>
void delete_device_memory(
    FPTYPE * &device) 
{
  if (device != NULL) {
    hipErrcheck(hipFree(device));
  }
}

template <typename FPTYPE>
void memset_device_memory(
    FPTYPE * device, 
    const FPTYPE var,
    const int size) 
{
  hipErrcheck(cudaMemset(device, var, sizeof(FPTYPE) * size));  
}
} // end of namespace deepmd