// zsthampi Zubin S Thampi
// sgarg7 Shaurya Garg
// kjadhav Karan Jadhav

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "mpi.h"

/* The number of grid points */
#define   NGRID			1000
/* first grid point */
#define   XI			-1.0
/* last grid point */
#define   XF			1.5
/* the value of epsilon */
#define EPSILON			0.005	

/* function declarations */
void        print_function_data(int, double*, double*, double*, int, int);
void        print_error_data(int np, double, double, double*, double*, double*, int numproc, int, int);
int         main(int, char**);
double      run(int,int,int,int);

/* returns the function y(x) = fn */
double fn(double x)
{
	return pow(x, 3) - pow(x,2) - x + 1;
	/* return pow(x, 2); */
	/* return x; */
}

/* returns the derivative d(fn)/dx = dy/dx */
double dfn(double x)
{
	return (3*pow(x,2)) - (2*x) - 1;
	/* return (2 * x); */
	/* return 1; */
}

int degreefn()
{
	return 3;
	/* return 2 */
	/* return 1 */
}

int main (int argc, char *argv[])
{
	/* Process Info */
	int numproc,rank; 

	/* MPI Initialize */
	MPI_Init(&argc,&argv);

	/* Get the number of processes - MPI_Comm_size */
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);

	/* Get the rank of the process - MPI_Comm_rank  */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Call the program
	// Sync all processes first
	MPI_Barrier(MPI_COMM_WORLD);
	// NON-BLOCKING COMMUNICATION AND MANUAL REDUCTION
	run(numproc,rank,0,0);
	// Sync all processes first
	MPI_Barrier(MPI_COMM_WORLD);
	// NON-BLOCKING COMMUNICATION AND MPI REDUCTION
	run(numproc,rank,0,1);
	// Sync all processes first
	MPI_Barrier(MPI_COMM_WORLD);
	// BLOCKING COMMUNICATION AND MANUAL REDUCTION
	run(numproc,rank,1,0);
	// Sync all processes first
	MPI_Barrier(MPI_COMM_WORLD);
	// BLOCKING COMMUNICATION AND MPI REDUCTION
	run(numproc,rank,1,1);

	/* MPI Finalize */
	MPI_Finalize();

	return 0;
}

// function to execute program with different communication and reduction methods
// block = 0 - NON-BLOCKING COMMUNICATION
// block = 1 - BLOCKING COMMUNICATION
// reduce = 0 - MANUAL REDUCTION
// reduce = 1 - MPI REDUCTION
double run(int numproc,int rank,int block,int reduce) {
	// Variables to store times taken for specific tasks, and overall time
	// Check time for overall program execution
	double start, end;
	// Check time for Finite Differencing step (which depends on communication method)
	double startFiniteDifferencing, endFiniteDifferencing;
	// Check time for Error computation step (which does not depend on communication method or reduction method)
	double startError, endError;
	// Check time for Reduction step (which depends on communication method and reduction method)
	double startReduce, endReduce;

	start = MPI_Wtime();

	double	*y, *dy;
	double	*err;

	// Initialize a local_min_max array of size ((degreefn()-1)*numproc)
	// Each process will only update (degreefn()-1) part of its array 

	// For instance, if there are 2 expected local minima/maxima and 4 processes
	// local_min_max will be an array of size 8 
	// process 0 will only update indices 0,1
	// process 1 will only update indices 2,3 . . . 

	// Finally, MPI_Reduce or Manual Reduction will combine all the local minima/maxima to array all_local_min_max
	double	local_min_max[(degreefn()-1)*numproc]; 
	double	all_local_min_max[(degreefn()-1)*numproc]; 

	// Variables for storing requests and statuses
	int request_size;
	MPI_Status status;
	MPI_Request request[4];
	MPI_Status request_status[4];
	MPI_Request next_request[numproc*5];
	MPI_Status next_request_status[numproc*5];

	// Start timer for Finite Differencing (From when we initialize x values)
	startFiniteDifferencing = MPI_Wtime();

	int size;
	if (rank==(numproc-1)) {
		/* For the last process, get size of remaining items */
		size = NGRID - ((NGRID/numproc)*(numproc-1));
	} else {
		/* For other processes, divide number of grids by (numproc-1) */
		size = (NGRID/numproc);
	}
	double x[size+2], dx;

	/* Compute x,y */
	for (int i=1; i<=size; i++) {
		x[i] = XI + (XF - XI) * (double)(i - 1 + ((NGRID/numproc)*rank))/(double)(NGRID - 1);
	}

	// Compute step size
	dx = x[2] - x[1];

	// Allocate function arrays
	y  =   (double*) malloc((size + 2) * sizeof(double));
	dy =   (double*) malloc((size + 2) * sizeof(double));

	//define the function
	for(int i = 1; i <= size; i++ )
	{
		y[i] = fn(x[i]);
	}

	/* MPI Send/Receive y values from neighbouring processes */
	/* Uses either BLOCKING or NON-BLOCKING communication */
	// Also set boundary values for first and last process
	// tag 0 -> leftmost element of process
	// tag 1 -> rightmost element of process

	if (rank==0) { 
		request_size = 2;
		// Boundary value for process 0
		y[0] = fn(x[1] - dx);
		if (numproc>1) {
			if (block) {
				MPI_Send(&y[size],1,MPI_DOUBLE,(rank+1),1,MPI_COMM_WORLD);
				MPI_Recv(&y[size+1],1,MPI_DOUBLE,(rank+1),0,MPI_COMM_WORLD,&status);
			} else {
				MPI_Isend(&y[size],1,MPI_DOUBLE,(rank+1),1,MPI_COMM_WORLD,&request[0]);
				MPI_Irecv(&y[size+1],1,MPI_DOUBLE,(rank+1),0,MPI_COMM_WORLD,&request[1]);
			}
		} else {
			// Set boundary value if there is only 1 process (Sequential)
			y[size+1] = fn(x[size] + dx);
		}
	} else if (rank==(numproc-1)) {
		request_size = 2;
		// Boundary value for last process
		y[size + 1] = fn(x[size] + dx);
		if (block) {
			MPI_Send(&y[1],1,MPI_DOUBLE,(rank-1),0,MPI_COMM_WORLD);
			MPI_Recv(&y[0],1,MPI_DOUBLE,(rank-1),1,MPI_COMM_WORLD,&status);
		} else {
			MPI_Isend(&y[1],1,MPI_DOUBLE,(rank-1),0,MPI_COMM_WORLD,&request[0]);
			MPI_Irecv(&y[0],1,MPI_DOUBLE,(rank-1),1,MPI_COMM_WORLD,&request[1]);
		}
	} else {
		request_size = 4;
		if (block) {
			MPI_Send(&y[1],1,MPI_DOUBLE,(rank-1),0,MPI_COMM_WORLD);
			MPI_Send(&y[size],1,MPI_DOUBLE,(rank+1),1,MPI_COMM_WORLD);
			MPI_Recv(&y[size+1],1,MPI_DOUBLE,(rank+1),0,MPI_COMM_WORLD,&status);
			MPI_Recv(&y[0],1,MPI_DOUBLE,(rank-1),1,MPI_COMM_WORLD,&status);
		} else {
			MPI_Isend(&y[1],1,MPI_DOUBLE,(rank-1),0,MPI_COMM_WORLD,&request[0]);
			MPI_Isend(&y[size],1,MPI_DOUBLE,(rank+1),1,MPI_COMM_WORLD,&request[1]);
			MPI_Irecv(&y[size+1],1,MPI_DOUBLE,(rank+1),0,MPI_COMM_WORLD,&request[2]);
			MPI_Irecv(&y[0],1,MPI_DOUBLE,(rank-1),1,MPI_COMM_WORLD,&request[3]);
		}
	}

	if (!block && numproc>1) {
		MPI_Waitall(request_size,request,request_status);
	}

	/* Compute dy(derivative) and dy(finite differencing) */
	for (int i = 1; i <= size; i++)
	{
		dy[i] = (y[i + 1] - y[i - 1])/(2.0 * dx);
	}

	// End timer for Finite Differencing
	endFiniteDifferencing = MPI_Wtime();
	printf("Rank %d : Finite Differencing Time %12.12lf : Block %d : Reduce %d \n",rank,endFiniteDifferencing - startFiniteDifferencing,block,reduce);

	// Start timer for Error Computation
	startError = MPI_Wtime();

	/* Compute errors in an array */
	err = (double*)malloc(size * sizeof(double));
	for(int i = 1; i <= size; i++)
	{
		err[i-1] = fabs( dy[i] - dfn(x[i]) );
	}

	// End timer for Error Computation
	endError = MPI_Wtime();
	printf("Rank %d : Error Computation Time %12.12lf : Block %d : Reduce %d \n",rank,endError - startError,block,reduce);

	/* Compute local minima/maxima and store it in an array */
	// Initialize local min/max to dummy values
	for(int i=0; i<((degreefn()-1)*numproc); i++)
	{
		local_min_max[i]=INT_MAX;
	}

	//find the local minima/maxima
	int count = 0;
	for(int i = 1; i <= size; i++)
	{
		if(fabs(dy[i]) < EPSILON)
		{
			if(count >= degreefn()-1)
			{
				printf("Warning: You have detected more than the maximum possible local minima/maxima.\n");
				printf("Ensure that DEGREE is accurate or reduce your EPSILON.\n");
				printf("Reseting count to zero.\n");
				count = 0;
			}
			// Each process only updates its section on the local_min_max array. 
			// For instance, if there are 2 expected local minima/maxima and 4 processes
			// local_min_max will be an array of size 8 
			// process 0 will only update indices 0,1
			// process 1 will only update indices 2,3 . . . 
			local_min_max[((degreefn()-1)*rank) + count] = x[i];
			count += 1;
		}
	}

	// Start timer for Reduce operation
	startReduce = MPI_Wtime();

	if (reduce) {
		/* MPI Reduce local minima/maxima to process 0 */
		MPI_Reduce(&local_min_max,&all_local_min_max,(degreefn()-1)*numproc,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
	}

	if (rank!=0) {
		/* MPI Send error array to process 0 */ 
		if (block) {
			MPI_Send(&err[0],size,MPI_DOUBLE,0,2,MPI_COMM_WORLD);
			MPI_Send(&x[1],size,MPI_DOUBLE,0,3,MPI_COMM_WORLD);
			MPI_Send(&y[1],size,MPI_DOUBLE,0,4,MPI_COMM_WORLD);
			MPI_Send(&dy[1],size,MPI_DOUBLE,0,5,MPI_COMM_WORLD);
		} else {
			MPI_Isend(&err[0],size,MPI_DOUBLE,0,2,MPI_COMM_WORLD,&next_request[0]);
			MPI_Isend(&x[1],size,MPI_DOUBLE,0,3,MPI_COMM_WORLD,&next_request[1]);
			MPI_Isend(&y[1],size,MPI_DOUBLE,0,4,MPI_COMM_WORLD,&next_request[2]);
			MPI_Isend(&dy[1],size,MPI_DOUBLE,0,5,MPI_COMM_WORLD,&next_request[3]);
		}

		if (!reduce) {
			if (block) {
				// MPI Send local minima/maxima to process 0 
				MPI_Send(&local_min_max[rank*(degreefn()-1)],(degreefn()-1),MPI_DOUBLE,0,6,MPI_COMM_WORLD);	
			} else {
				MPI_Isend(&local_min_max[rank*(degreefn()-1)],(degreefn()-1),MPI_DOUBLE,0,6,MPI_COMM_WORLD,&next_request[4]);	
			}
		}

		if (!block) {
			if (reduce) {
				MPI_Waitall(4,next_request,next_request_status);
			} else {
				MPI_Waitall(5,next_request,next_request_status);
			}
		}

		// End timer for Reduce operation
		endReduce = MPI_Wtime();
		printf("Rank %d : Reduce Operation Time %12.12lf : Block %d : Reduce %d \n",rank,endReduce - startReduce,block,reduce);
	}


	/* 
	 * Process 0 will collect data and compute mean and sd
	 * All processes will be used for computation
	 */

	if (rank==0) {
		// Variables to store collected values
		// double all_err[NGRID];
		// double all_x[NGRID];
		// double all_y[NGRID];
		// double all_dy[NGRID];
		double	*all_err, *all_x, *all_y, *all_dy;
		all_err  =   (double*) malloc((NGRID) * sizeof(double));
		all_x  =   (double*) malloc((NGRID) * sizeof(double));
		all_y  =   (double*) malloc((NGRID) * sizeof(double));
		all_dy  =   (double*) malloc((NGRID) * sizeof(double));

		/* MPI Receive error,x,y and dy values from all other processes */
		int count;
		for (count=0; count<size; count++) {
			all_err[count] = err[count];
			all_x[count] = x[count+1];
			all_y[count] = y[count+1];
			all_dy[count] = dy[count+1];
		}

		if (numproc>1) {
			for (int i=1; i<numproc; i++) {
				if (block) {
					MPI_Recv(&all_err[count],(NGRID - count),MPI_DOUBLE,i,2,MPI_COMM_WORLD,&status);
					MPI_Recv(&all_x[count],(NGRID - count),MPI_DOUBLE,i,3,MPI_COMM_WORLD,&status);
					MPI_Recv(&all_y[count],(NGRID - count),MPI_DOUBLE,i,4,MPI_COMM_WORLD,&status);
					MPI_Recv(&all_dy[count],(NGRID - count),MPI_DOUBLE,i,5,MPI_COMM_WORLD,&status);
				} else {
					MPI_Irecv(&all_err[count],(NGRID - count),MPI_DOUBLE,i,2,MPI_COMM_WORLD,&next_request[((i-1)*4)]);
					MPI_Irecv(&all_x[count],(NGRID - count),MPI_DOUBLE,i,3,MPI_COMM_WORLD,&next_request[((i-1)*4)+1]);
					MPI_Irecv(&all_y[count],(NGRID - count),MPI_DOUBLE,i,4,MPI_COMM_WORLD,&next_request[((i-1)*4)+2]);
					MPI_Irecv(&all_dy[count],(NGRID - count),MPI_DOUBLE,i,5,MPI_COMM_WORLD,&next_request[((i-1)*4)+3]);
				}
				count += (NGRID/numproc);
			}
		}

		/* Receive local minima/maxima and combine manually */
		if (!reduce) {
			for (int i=0; i<(degreefn()-1); i++) {
				all_local_min_max[i] = local_min_max[i];
			}
			if (numproc>1) {
				if (block) {
					for (int i=1; i<numproc; i++) {
						MPI_Recv(&all_local_min_max[(degreefn()-1)*i],(degreefn()-1),MPI_DOUBLE,i,6,MPI_COMM_WORLD,&status);
					}
				} else {
					for (int i=1; i<numproc; i++) {
						MPI_Irecv(&all_local_min_max[(degreefn()-1)*i],(degreefn()-1),MPI_DOUBLE,i,6,MPI_COMM_WORLD,&next_request[((numproc-1)*4)+(i-1)]);
					}
				}
			}
		}

		// Wait for non-blocking communication, if present
		if (!block) {
			if (reduce) {
				MPI_Waitall((numproc-1)*4,next_request,next_request_status);
			} else {
				MPI_Waitall(((numproc-1)*5),next_request,next_request_status);
			}
		}

		// End timer for Reduce operation
		endReduce = MPI_Wtime();
		printf("Rank %d : Reduce Operation Time %12.12lf : Block %d : Reduce %d \n",rank,endReduce - startReduce,block,reduce);

		// After combining all minima and maxima, check whether the count exceeded the expected number of local minima/maxima
		count = 0;
		for (int i=0; i<((degreefn()-1)*numproc); i++) {
			if (all_local_min_max[i] != INT_MAX) {
				count += 1;
			}
		}
		// Print Warning if the count exceed the expected number of local minima/maxima
		if(count > degreefn()-1) {
			printf("Warning: You have detected more than the maximum possible local minima/maxima.\n");
			printf("Ensure that DEGREE is accurate or reduce your EPSILON.\n");
		}

		/* Compute Mean and SD */
		//find the average error
		double avg_err = 0.0;
		for(int i = 0; i < NGRID ; i++)
			avg_err += all_err[i];

		avg_err /= (double)NGRID;

		//find the standard deviation of the error
		double std_dev = 0.0;
		for(int i = 0; i< NGRID; i++)
		{
			std_dev += pow(all_err[i] - avg_err, 2);
		}
		std_dev = sqrt(std_dev/(double)NGRID);

		/* Print */
		print_function_data(NGRID, &all_x[0], &all_y[0], &all_dy[0],block,reduce);
		print_error_data(NGRID, avg_err, std_dev, &all_x[0], &all_err[0], all_local_min_max, numproc,block,reduce);

		free(all_err);
		free(all_x);
		free(all_y);
		free(all_dy);
	} 

	//free allocated memory 
	free(y);
	free(dy);
	free(err);

	end = MPI_Wtime();
	printf("Rank %d : Total Time %12.12lf : Block %d : Reduce %d \n",rank,end - start,block,reduce);
	return (end-start);
}

//prints out the function and its derivative to a file
void print_function_data(int np, double *x, double *y, double *dydx, int block, int reduce)
{
	int   i;
	char file_name[80];
	strcpy(file_name, "fn");
	if(block){
	strcat(file_name, "_blk");
	}
	if(reduce){
	strcat(file_name, "_red");
	}
	strcat(file_name, ".dat");
	FILE *fp = fopen(file_name, "w");

	for(i = 0; i < np; i++)
	{
		fprintf(fp, "%f %f %f\n", x[i], y[i], dydx[i]);
	}
	fclose(fp);
}

void print_error_data(int np, double avgerr, double stdd, double *x, double *err, double *local_min_max, int numproc, int block, int reduce)
{
	int   i;
	char file_name[80];
	strcpy(file_name, "err");
	if(block){
	strcat(file_name, "_blk");
	}
	if(reduce){
	strcat(file_name, "_red");
	}
	strcat(file_name, ".dat");
	FILE *fp = fopen(file_name, "w");

	fprintf(fp, "%e\n%e\n", avgerr, stdd);

	int count = 0;
	for(i = 0; i<(degreefn()-1)*numproc; i++)
	{
		if (local_min_max[i] != INT_MAX && count<(degreefn()-1)) {
			fprintf(fp, "(%f, %f)\n", local_min_max[i], fn(local_min_max[i]));
			count += 1;
		}
	}
	while (count<(degreefn()-1)) {
		fprintf(fp, "(UNDEF, UNDEF)\n");
		count += 1;
	}

	for(i = 0; i < np; i++)
	{
		fprintf(fp, "%f %e \n", x[i], err[i]);
	}
	fclose(fp);
}

