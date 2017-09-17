// zsthampi Zubin S Thampi
// sgarg7 Shaurya Garg
// kjadhav Karan Jadhav

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include "mpi.h"
#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

extern int tpdt(double *t, double dt, double end_time);
extern int nproc, rank;

// Define values. Must be the same as defined in lake_mpi.cu
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

// Function to evolve data (n ==> n_with_buffer)
__global__ void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t, int *nblocks,int rank)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = (i+2)*n + (j+2);
	int n_wob = n-4;
	if( rank==0 && (i <=1  || j >= n_wob - 2 ))
	{
		un[idx] = 0.;
	}
	else if( rank==1 && (i >= n_wob - 2 || j >= n_wob - 2 ))
	{
		un[idx] = 0.;
	}
	else if( rank==2 && (i <=1 || j <= 1 ))
	{
		un[idx] = 0.;
	}
	else if( rank==3 && (i >= n_wob-2 || j <= 1 ))
	{
		un[idx] = 0.;
	}
	else
	{
		un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + uc[idx + n] + uc[idx - n] + 
					0.25*(uc[idx-n-1] + uc[idx-n+1] + uc[idx+n-1] + uc[idx+n+1]) + 
					0.125*(uc[idx-2] + uc[idx+2] + uc[idx+2*n] + uc[idx-2*n]) - 6 * uc[idx])/(h * h) + (-expf(-TSCALE * t) * pebbles[idx]));
	}
	__syncthreads();
}

// Function to copy horizontal data
void copy_horizontal(double *dest,double *src,int n_with_buffer,int rank)
{
	int j;
	switch(rank){
		case 0:
			j = 2;
			break;
		case 1:
			j = 2;
			break;
		case 2:
			j = n_with_buffer -4;
			break;
		case 3:
			j = n_with_buffer -4;
			break;
		default:
			printf("Invalid value in copy_horizontal\n" );
	}
	//Go to appropriate row
	int index = 2*(n_with_buffer);
	//Go to proper index
	index +=j;
	//Copy row by row (2 elements) in src
	int iter=0;
	while(iter<n_with_buffer-4){
		cudaMemcpy(&dest[2*iter], &src[index], sizeof(double) * (2), cudaMemcpyDeviceToHost);
		index +=n_with_buffer;
		iter++;
	}

}

// Function to copy vertical data
void copy_vertical(double* dest,double *src,int n_with_buffer, int rank){
	int i;
	switch(rank){
		case 0:
			i = n_with_buffer -4;
			break;
		case 1:
			i = 2;
			break;
		case 2:
			i = n_with_buffer -4;
			break;
		case 3:
			i = 2;
			break;
		default:
			printf("Invalid value in copy_horizontal\n" );
	}
	//Go to appropriate row
	int index = i*(n_with_buffer);
	//Go to proper index
	index +=2;
	//Copy first row (n elements) in src
	cudaMemcpy(dest, &src[index], sizeof(double) * (n_with_buffer-4), cudaMemcpyDeviceToHost);
	//Copy second row (n elements) in src
	index +=n_with_buffer;
	cudaMemcpy(&dest[n_with_buffer-4], &src[index], sizeof(double) * (n_with_buffer-4), cudaMemcpyDeviceToHost);
}

// Function to copy diagonal data
void copy_diagonal(double* dest,double *src,int n_with_buffer, int rank){
	int i,j;
	switch(rank){
		case 0:
			i= n_with_buffer -3;
			j= 2;
			break;
		case 1:
			i= 2;
			j= 2;
			break;
		case 2:
			i= n_with_buffer-3;
			j= n_with_buffer -3;
			break;
		case 3:
			i= 2;
			j= n_with_buffer-3;
	}
	int index = (i*n_with_buffer)+j;
	cudaMemcpy(dest, &src[index], sizeof(double), cudaMemcpyDeviceToHost);

}

// Function to make MPI_Recv calls
void recvData(double *vertical_recv,double *horizontal_recv, double *diagonal_recv,int n_with_buffer, int rank, MPI_Request *requests)
{
	int n = n_with_buffer-4;
	MPI_Status receive_status;
	switch(rank){
		case 0:
			MPI_Irecv(vertical_recv, 2*n, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &requests[3]);
			MPI_Irecv(horizontal_recv, 2*n, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, &requests[4]);
			MPI_Irecv(diagonal_recv, 2*n, MPI_DOUBLE, 3, 3, MPI_COMM_WORLD, &requests[5]);
			break;
		case 1:
			MPI_Irecv(vertical_recv, 2*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requests[3]);
			MPI_Irecv(horizontal_recv, 2*n, MPI_DOUBLE, 3, 3, MPI_COMM_WORLD, &requests[4]);
			MPI_Irecv(diagonal_recv, 2*n, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, &requests[5]);
			break;
		case 2:
			MPI_Irecv(vertical_recv, 2*n, MPI_DOUBLE, 3, 3, MPI_COMM_WORLD, &requests[3]);
			MPI_Irecv(horizontal_recv, 2*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requests[4]);
			MPI_Irecv(diagonal_recv, 2*n, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &requests[5]);
			break;
		case 3:
			MPI_Irecv(vertical_recv, 2*n, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, &requests[3]);
			MPI_Irecv(horizontal_recv, 2*n, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &requests[4]);
			MPI_Irecv(diagonal_recv, 2*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requests[5]);
			break;

	}


}

// Function to make MPI Send calls
void sendData(double *vertical,double *horizontal, double *diagonal,int n_with_buffer, int rank,MPI_Request *requests)
{
	int n = n_with_buffer-4;
	switch(rank){
		case 0:
			MPI_Isend(vertical, 2*n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,&requests[0]);	
			MPI_Isend(horizontal, 2*n, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD,&requests[1]);	
			MPI_Isend(diagonal, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD,&requests[2]);	
			break;
		case 1:
			MPI_Isend(vertical, 2*n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,&requests[0]);	
			MPI_Isend(horizontal, 2*n, MPI_DOUBLE, 3, 1, MPI_COMM_WORLD,&requests[1]);	
			MPI_Isend(diagonal, 1, MPI_DOUBLE, 2, 1, MPI_COMM_WORLD,&requests[2]);	
			break;
		case 2:
			MPI_Isend(vertical, 2*n, MPI_DOUBLE, 3, 2, MPI_COMM_WORLD,&requests[0]);	
			MPI_Isend(horizontal, 2*n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD,&requests[1]);	
			MPI_Isend(diagonal, 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD,&requests[2]);	
			break;
		case 3:
			MPI_Isend(vertical, 2*n, MPI_DOUBLE, 2, 3, MPI_COMM_WORLD,&requests[0]);	
			MPI_Isend(horizontal, 2*n, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD,&requests[1]);	
			MPI_Isend(diagonal, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD,&requests[2]);	
			break;

	}


}

// Function to paste diagonal data
void paste_diagonal(double* dest,double *src,int n_with_buffer, int rank) 
{
	int i,j;
	switch(rank){
		case 0:
			i= n_with_buffer -2;
			j= 2;
			break;
		case 1:
			i= 1;
			j= 1;
			break;
		case 2:
			i= n_with_buffer -2;
			j= n_with_buffer -2;
			break;
		case 3:
			i= 1;
			j= n_with_buffer -2;
	}
	int index = (i*n_with_buffer)+j;
	cudaMemcpy(&dest[index], src, sizeof(double), cudaMemcpyHostToDevice);

}

// Function to paste vertical data
void paste_vertical(double* dest,double *src,int n_with_buffer, int rank){
	int i;
	switch(rank){
		case 0:
			i = n_with_buffer -2;
			break;
		case 1:
			i = 0;
			break;
		case 2:
			i = n_with_buffer -2;
			break;
		case 3:
			i = 0;
			break;
		default:
			printf("Invalid value in copy_horizontal\n" );
	}
	//Go to appropriate row
	int index = i*(n_with_buffer);
	//Go to proper index
	index +=2;
	//Copy first column (n elements) in src
	cudaMemcpy(&dest[index], src, sizeof(double) * (n_with_buffer-4), cudaMemcpyHostToDevice);
	//Copy second column (n elements) in src
	index +=n_with_buffer;
	cudaMemcpy(&dest[index], &src[n_with_buffer-4], sizeof(double) * (n_with_buffer-4), cudaMemcpyHostToDevice);
}

// Function to paste horizontal data
void paste_horizontal(double *dest,double *src,int n_with_buffer,int rank)
{
	int j;
	switch(rank){
		case 0:
			j = 0;
			break;
		case 1:
			j = 0;
			break;
		case 2:
			j = n_with_buffer -2;
			break;
		case 3:
			j = n_with_buffer -2;
			break;
		default:
			printf("Invalid value in copy_horizontal\n" );
	}
	//Go to appropriate row
	int index = 2*(n_with_buffer);
	//Go to proper index
	index +=j;
	//Copy row by row (2 elements) in src
	int iter=0;
	while(iter<n_with_buffer-4){
		cudaMemcpy(&dest[index], &src[2*iter], sizeof(double) * (2), cudaMemcpyHostToDevice);
		index +=n_with_buffer;
		iter++;
	}

}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads,int rank, int start_i, int start_j)
{
	cudaEvent_t kstart, kstop;
	float ktime;
	double t, dt;

	// Add size 4 as buffer to n value - 2 rows and columns each 
	int n_with_buffer = n+4;

	/* HW2: Define your local variables here */

	int nblocks = n/nthreads;
	int *nblocks_d;
	double *u_d;
	double *u0_d;
	double *u1_d;
	double *pebbles_d;


	// We split up the boundary values into 3 types - HORIZONTAL, VERTICAL and DIAGONAL
	// Each process will have 1 of each to communicate to a neighbouring process 
	// DIAGONAL will only be 1 value
	// HORIZONTAL and VERITICAL will have 2*n values

	double *diagonal,*horizontal,*vertical,*diagonal_recv,*horizontal_recv,*vertical_recv;
	diagonal = (double*)malloc(sizeof(double));
	horizontal = (double*)malloc(sizeof(double) * (2*n));
	vertical = (double*)malloc(sizeof(double) * (2*n));
	diagonal_recv = (double*)malloc(sizeof(double));
	horizontal_recv = (double*)malloc(sizeof(double) * (2*n));
	vertical_recv = (double*)malloc(sizeof(double) * (2*n));
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

	// Copy pebbles and u values from Host to Device 
	cudaMemcpy(nblocks_d, &nblocks, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(u0_d, u0, sizeof(double) * n_with_buffer * n_with_buffer, cudaMemcpyHostToDevice);
	cudaMemcpy(u1_d, u1, sizeof(double) * n_with_buffer * n_with_buffer, cudaMemcpyHostToDevice);
	cudaMemcpy(pebbles_d, pebbles, sizeof(double) * n_with_buffer * n_with_buffer, cudaMemcpyHostToDevice);


	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

	/* HW2: Add main lake simulation loop here */

	t = 0.;
	dt = h / 2.;

	while(1)
	{
		// Communicate boundary values first
		MPI_Status request_status[6];
		MPI_Request requests[6];
		// Copy boundary values at each rank
		copy_horizontal(horizontal,u1_d,n_with_buffer,rank);
		copy_vertical(vertical,u1_d,n_with_buffer,rank);
		copy_diagonal(diagonal,u1_d,n_with_buffer,rank);
		// Send the boundary values to respective processes
		sendData(vertical,horizontal,diagonal,n_with_buffer,rank,requests);
		// Receive data at respective processes
		recvData(vertical_recv,horizontal_recv,diagonal_recv,n_with_buffer,rank,requests);
		// Wait for all communication to end
		MPI_Waitall(6,requests,request_status);
		// Paste the boundary values at the received processes
		paste_horizontal(u1_d,horizontal_recv,n_with_buffer,rank);
		paste_vertical(u1_d,vertical_recv,n_with_buffer,rank);
		paste_diagonal(u1_d,diagonal_recv,n_with_buffer,rank);

		evolve<<< block,thread >>>(u_d, u1_d, u0_d, pebbles_d, n_with_buffer, h, dt, t, nblocks_d,rank);

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
	free(diagonal);
	free(horizontal);
	free(vertical);
	free(diagonal_recv);
	free(horizontal_recv);
	free(vertical_recv);
}
