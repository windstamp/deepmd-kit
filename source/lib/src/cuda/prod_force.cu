#include "device.h"

#if GOOGLE_CUDA
#include "gpu_cuda.h"
#elif PADDLE_HIP
#include "gpu_hip.h"
#endif

#include "prod_force.h"

template <
    typename FPTYPE,
    int      THREADS_PER_BLOCK>
__global__ void force_deriv_wrt_center_atom(
    FPTYPE * force, 
    const FPTYPE * net_deriv, 
    const FPTYPE * in_deriv, 
    const int ndescrpt)
{
  __shared__ FPTYPE data[THREADS_PER_BLOCK * 3];
  unsigned int bid = blockIdx.x;
  unsigned int tid = threadIdx.x;
  for (int ii = tid; ii < THREADS_PER_BLOCK * 3; ii += THREADS_PER_BLOCK) {
    data[ii] = 0.f;
  }
  for (int ii = tid; ii < ndescrpt; ii += THREADS_PER_BLOCK) {
    for (int jj = 0; jj < 3; jj++) {
      data[jj * THREADS_PER_BLOCK + tid] += net_deriv[bid * ndescrpt + ii] * in_deriv[bid * ndescrpt * 3 + ii * 3 + jj];
    }
  }
  __syncthreads(); 
  // do reduction in shared memory
  for (int ii = THREADS_PER_BLOCK >> 1; ii > 0; ii >>= 1) {
    if (tid < ii) {
      for (int jj = 0; jj < 3; jj++) {
        data[jj * THREADS_PER_BLOCK + tid] += data[jj * THREADS_PER_BLOCK + tid + ii];
      }
    }
    __syncthreads();
  }
  // write result for this block to global memory
  if (tid == 0) {
    force[bid * 3 + 0] -= data[THREADS_PER_BLOCK * 0];
    force[bid * 3 + 1] -= data[THREADS_PER_BLOCK * 1];
    force[bid * 3 + 2] -= data[THREADS_PER_BLOCK * 2];
  }
}

template<typename FPTYPE>
__global__ void force_deriv_wrt_neighbors_a(
    FPTYPE * force, 
    const FPTYPE * net_deriv,
    const FPTYPE * in_deriv,
    const int * nlist,
    const int nloc,
    const int nnei)
{  
    // idy -> nnei
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idz = threadIdx.y;
    const unsigned int idw = threadIdx.z;
    const int ndescrpt = nnei * 4;
    if (idx >= nloc) {
        return;
    }
    // deriv wrt neighbors
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    atomicAdd(
        force + j_idx * 3 + idz, 
        net_deriv[idx * ndescrpt + idy * 4 + idw] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz]);
}

template<typename FPTYPE>
__global__ void force_deriv_wrt_neighbors_r(
		FPTYPE * force, 
		const FPTYPE * net_deriv,
		const FPTYPE * in_deriv,
		const int * nlist,
		const int nloc,
		const int nnei)
{  
    // idy -> nnei
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idz = threadIdx.y;
    const int ndescrpt = nnei * 1;
    if (idx >= nloc) {
        return;
    }
    // deriv wrt neighbors
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    atomicAdd(
        force + j_idx * 3 + idz, 
        net_deriv[idx * ndescrpt + idy] * in_deriv[idx * ndescrpt * 3 + idy * 3 + idz]);
}

namespace deepmd {
template<typename FPTYPE> 
void prod_force_a_gpu_cuda(    
    FPTYPE * force, 
    const FPTYPE * net_deriv, 
    const FPTYPE * in_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei)
{
  const int ndescrpt = nnei * 4;

  #if GOOGLE_CUDA
  cudaErrcheck(cudaMemset(
      force, 
      0.0, sizeof(FPTYPE) * nall * 3));
  #elif PADDLE_HIP
  hipErrcheck(hipMemset(
      force, 
      0.0, sizeof(FPTYPE) * nall * 3));
  #endif

  force_deriv_wrt_center_atom<FPTYPE, TPB> <<<nloc, TPB>>>(
      force, 
      net_deriv, in_deriv, ndescrpt);

  const int LEN = 64;
  const int nblock = (nloc + LEN -1) / LEN;
  dim3 block_grid(nblock, nnei);
  dim3 thread_grid(LEN, 3, 4);
  force_deriv_wrt_neighbors_a<<<block_grid, thread_grid>>>(
      force, 
      net_deriv, in_deriv, nlist, nloc, nnei);
}

template<typename FPTYPE> 
void prod_force_r_gpu_cuda(    
    FPTYPE * force, 
    const FPTYPE * net_deriv, 
    const FPTYPE * in_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei)
{
  const int ndescrpt = nnei * 1;

  #if GOOGLE_CUDA
  cudaErrcheck(cudaMemset(
      force, 
      0.0, sizeof(FPTYPE) * nall * 3));
  #elif PADDLE_HIP
  hipErrcheck(hipMemset(
      force, 
      0.0, sizeof(FPTYPE) * nall * 3));
  #endif

  #if GOOGLE_CUDA
  force_deriv_wrt_center_atom<FPTYPE, TPB> <<<nloc, TPB>>>(
      force, 
      net_deriv, in_deriv, ndescrpt);
  #elif PADDLE_HIP
  hipLaunchKernelGGL(force_deriv_wrt_center_atom<FPTYPE, TPB>, dim3(nloc), dim3(TPB), 0, 0,
      force, 
      net_deriv, in_deriv, ndescrpt);
  #endif

  const int LEN = 64;
  const int nblock = (nloc + LEN -1) / LEN;
  dim3 block_grid(nblock, nnei);
  dim3 thread_grid(LEN, 3);

  #if GOOGLE_CUDA
  force_deriv_wrt_neighbors_r<<<block_grid, thread_grid>>>(
      force, 
      net_deriv, in_deriv, nlist, nloc, nnei);
  #elif PADDLE_HIP
  hipLaunchKernelGGL(force_deriv_wrt_neighbors_r, block_grid, thread_grid, 0, 0,
      force, 
      net_deriv, in_deriv, nlist, nloc, nnei);
  #endif
}

template void prod_force_a_gpu_cuda<float>(float * force, const float * net_deriv, const float * in_deriv, const int * nlist, const int nloc, const int nall, const int nnei);
template void prod_force_a_gpu_cuda<double>(double * force, const double * net_deriv, const double * in_deriv, const int * nlist, const int nloc, const int nall, const int nnei);
template void prod_force_r_gpu_cuda<float>(float * force, const float * net_deriv, const float * in_deriv, const int * nlist, const int nloc, const int nall, const int nnei);
template void prod_force_r_gpu_cuda<double>(double * force, const double * net_deriv, const double * in_deriv, const int * nlist, const int nloc, const int nall, const int nnei);
}
