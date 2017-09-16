#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

extern int tpdt(double *t, double dt, double end_time);
extern int nproc, rank;

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}

// Function to evolve data 
__global__ void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t, int *nblocks)
{
  int idx = (blockIdx.y * blockDim.y + threadIdx.y) * (blockDim.x * *nblocks) + (blockIdx.x * blockDim.x + threadIdx.x);
  int i = idx / n;
  int j = idx % n;

  if( i <= 1 || i >= n-2 || j <= 1 || j >= n - 2 )
  {
    un[idx] = 0.;
  }
  else
  {
    // 5 point stencil
    // un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + 
    //             uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) + f(pebbles[idx],t));

    // 13 point stencil
    un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx - n] + 
              0.25*(uc[idx-n-1] + uc[idx-n+1] + uc[idx+n-1] + uc[idx+n+1]) + 
              0.125*(uc[idx-2] + uc[idx+2] + uc[idx+2*n] + uc[idx-2*n]) - 6 * uc[idx])/(h * h) + (-expf(-TSCALE * t) * pebbles[idx]));
  }
  __syncthreads();
}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;
  double t, dt;

  int n_with_buffer = n+4;
        
	/* HW2: Define your local variables here */

  int nblocks = n/nthreads;
  int *nblocks_d;
  double *u_d;
  double *u0_d;
  double *u1_d;
  double *pebbles_d;

  dim3 block(nblocks,nblocks,1);
  dim3 thread(nthreads,nthreads,1);

  /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */

  cudaMalloc((void **) &nblocks_d, sizeof(int));
  cudaMalloc((void **) &u_d, sizeof(double) * n_with_buffer * n_with_buffer); 
  cudaMalloc((void **) &u0_d, sizeof(double) * n_with_buffer * n_with_buffer); 
  cudaMalloc((void **) &u1_d, sizeof(double) * n_with_buffer * n_with_buffer); 
  cudaMalloc((void **) &pebbles_d, sizeof(double) * n_with_buffer * n_with_buffer); 

  // Copy all three pointers from Host to Device 
  cudaMemcpy(nblocks_d, &nblocks, sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemset(u_d, 0, sizeof(double) * n * n);
  // cudaMemcpy(u_d, u, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(u0_d, u0, sizeof(double) * n_with_buffer * n_with_buffer, cudaMemcpyHostToDevice);
  cudaMemcpy(u1_d, u1, sizeof(double) * n_with_buffer * n_with_buffer, cudaMemcpyHostToDevice);
  cudaMemcpy(pebbles_d, pebbles, sizeof(double) * n_with_buffer * n_with_buffer, cudaMemcpyHostToDevice);
  

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

	/* HW2: Add main lake simulation loop here */

  // do while 
  // Check condition
  // evolve<<< block, thread >>>()
  // Copy within device to respective pointers
  t = 0.;
  dt = h / 2.;

  while(1)
  {
    // !!!TODO!!! Copy from Device to Host (u1) (Only the buffer!)
    // 3 arrays

    // !!!TODO!!! Communicate values using MPI
    // SEND & RECEIVE

    // cudaMemcpy only the buffer

    evolve<<< block,thread >>>(u_d, u1_d, u0_d, pebbles_d, n, h, dt, t, nblocks_d);

    // Copy updated pointers from Device to Device
    cudaMemcpy(u0_d, u1_d, sizeof(double) * n_with_buffer * n_with_buffer, cudaMemcpyDeviceToDevice);
    cudaMemcpy(u1_d, u_d, sizeof(double) * n_with_buffer * n_with_buffer, cudaMemcpyDeviceToDevice);

    if(!tpdt(&t,dt,end_time)) break;
  }

	
  /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */

  // Copy from Device to Host
  cudaMemcpy(u, u_d, sizeof(double) * n_with_buffer * n_with_buffer, cudaMemcpyDeviceToHost);

	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));

  cudaFree(nblocks_d);
  cudaFree(u_d);
  cudaFree(u0_d);
  cudaFree(u1_d);
  cudaFree(pebbles_d);
}
