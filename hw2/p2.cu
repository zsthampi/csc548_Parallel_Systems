// Author : zsthampi Zubin S Thampi
/* Program to compute Pi using the Leibniz formula */

#include <stdio.h>
#include <math.h>
#include "mytime.h"

#define THREADS 512
#define MAX_BLOCKS 16

__global__ void Leibniz(int *n, int *blocks, double *gsum) {
  __shared__ double sum[THREADS];
  // Variables to use for reduction process 
  int k, old_k;
  double power;

  // if (threadIdx.x==0) { printf("STARTING : Leibniz \n"); }
  
  sum[threadIdx.x] = 0.0;
  for (k = blockIdx.x * blockDim.x + threadIdx.x; k < *n; k += blockDim.x * *blocks) {
    if (k%2==0) {
      power = 1.0;
    } else {
      power = -1.0;
    }
    sum[threadIdx.x] += (power / (double)(2.0 * k + 1.0));
  }

  sum[threadIdx.x] *= 4.0;

  __syncthreads();

  // Block Reduction - old logic

  // for (k = blockDim.x/2; k > 0; k >>= 1) {
  //   if (threadIdx.x==0) { printf("k value : %d \n",k); }
  //   if (threadIdx.x < k) {
  //     sum[threadIdx.x] += sum[threadIdx.x + k];
  //   }
  //   __syncthreads();
  // }

  // Block Reduction - new logic
  // I basically changed the shift operator to divide
  // And while adding up, I check a boundary condition too
  // Thus it will counter imbalances caused due to THREADS not being a power of 2

  k = blockDim.x;
  while (k>1) {
    old_k = k;
    if (k%2==0) {
      k = k/2;
    } else {
      k = (k+1)/2;
    }
    // if (threadIdx.x==0) { printf("k value : %d \n",k); }
    if (threadIdx.x < k && (threadIdx.x + k) < old_k) {
      sum[threadIdx.x] += sum[threadIdx.x + k];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 )
    gsum[blockIdx.x] = sum[threadIdx.x];
}

__global__ void global_reduce(int *n, int *blocks, double *gsum) {
  int k, old_k;
  __shared__ double sum[THREADS];
  // if (threadIdx.x==0) { printf("STARTING : Global Reduce \n"); }

  sum[threadIdx.x] = gsum[threadIdx.x];
  __syncthreads();

  // Block Reduction - old logic

  // for (k = blockDim.x/2; k > 0 ; k >>= 1) {
  //   if (threadIdx.x==0) { printf("k value : %d \n",k); }
  //   if (threadIdx.x < k) {
  //     sum[threadIdx.x] += sum[threadIdx.x + k];
  //   }
  //   __syncthreads();
  // }

  // Block Reduction - new logic
  // I basically changed the shift operator to divide
  // And while adding up, I check a boundary condition too
  // Thus it will counter imbalances caused due to THREADS not being a power of 2

  k = blockDim.x;
  while (k>1) {
    old_k = k;
    if (k%2==0) {
      k = k/2;
    } else {
      k = (k+1)/2;
    }
    // if (threadIdx.x==0) { printf("k value : %d \n",k); }

    if (threadIdx.x < k && (threadIdx.x + k) < old_k) {
      sum[threadIdx.x] += sum[threadIdx.x + k];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
    gsum[threadIdx.x] = sum[threadIdx.x];
}

int main(int argc, char *argv[]) {
  int n;
  int blocks = MAX_BLOCKS;
  // Device copy of number of intervals
  int *n_d, *blocks_d;
  double PI25DT = 3.141592653589793238462643;
  double pi;
  // double mypi[THREADS * blocks];
  // Device copy of computed pi value
  double *mypi_d;
  struct timeval startwtime, endwtime, diffwtime;
  
  // Allocate memory
  cudaMalloc((void **) &n_d, sizeof(int));
  cudaMalloc((void **) &blocks_d, sizeof(int));
  cudaMalloc((void **) &mypi_d, sizeof(double) * blocks);

  while (1) {
    printf("Enter the number of intervals: (0 quits) ");fflush(stdout);
    scanf("%d",&n);

    gettimeofday(&startwtime, NULL);
    if (n == 0)
      break;

    // Copy from Host to Device
    cudaMemcpy(n_d,&n,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(blocks_d,&blocks,sizeof(int),cudaMemcpyHostToDevice);

    Leibniz<<< blocks,THREADS >>>(n_d,blocks_d,mypi_d);

    // Copy from Device to Host
    // cudaMemcpy(&mypi,mypi_d,sizeof(double) * THREADS * blocks,cudaMemcpyDeviceToHost);

    global_reduce<<< 1,blocks >>>(n_d,blocks_d,mypi_d);
    cudaMemcpy(&pi,mypi_d,sizeof(double),cudaMemcpyDeviceToHost);

    // cudaMemcpy(&mypi,mypi_d,sizeof(double) * blocks,cudaMemcpyDeviceToHost);
    // for (int i=0; i<blocks; i++) {
    //   printf("%d : %lf \n",i,mypi[i]);
    // }

    // pi = 0.0;
    // for (int i=0; i<THREADS; i++) {
    //   pi += mypi[i];
    // }

    gettimeofday(&endwtime, NULL);
    MINUS_UTIME(diffwtime, endwtime, startwtime);
    printf("pi is approximately %.16f, Error is %.16f\n",
	   pi, fabs(pi - PI25DT));
    printf("wall clock time = %d.%06d\n",
	   diffwtime.tv_sec, diffwtime.tv_usec);
  }

  // Free memory
  cudaFree(n_d);
  cudaFree(blocks_d);
  cudaFree(mypi_d);
  return 0;
}
