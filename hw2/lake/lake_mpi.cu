// zsthampi Zubin S Thampi
// sgarg7 Shaurya Garg
// kjadhav Karan Jadhav

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

void stitch(double *u,double *u_proc,double *u_proc1,double *u_proc2,double *u_proc3,int npoints_proc);
void init(double *u, double *pebbles_proc, int npoints_proc);
void init_original(double *u, double *pebbles_proc, int npoints);
int tpdt(double *t, double dt, double end_time);
void print_heatmap_original(const char *filename, double *u, int n, double h);
void print_heatmap(const char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);
void divide(double *pebbles_proc, double *pebbles, int npoints_proc, int start_i,int start_j);


extern void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads, int rank, int start_i, int start_j);

int main(int argc, char *argv[])
{
	// Set buffer to print all statement immediately
	setbuf(stdout, NULL);

	// Variables to store MPI size and rank
	int nproc, rank;

	MPI_Init(&argc,&argv);

	// Number of processes will always be 4 for this problem
	MPI_Comm_size(MPI_COMM_WORLD,&nproc);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	if(argc != 5)
	{
		printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
		return 0;
	}

	int     npoints   = atoi(argv[1]);
	// The number of points inside each process( where nproc will always be 4 )
	int     npoints_proc = npoints / sqrt(nproc);
	int     npebs     = atoi(argv[2]);
	double  end_time  = (double)atof(argv[3]);
	int     nthreads  = atoi(argv[4]);
	// printf("%d : %d : %d : %lf : %d \n",rank,npoints,npebs,end_time,nthreads);

	// NAMING CONVENTION 
	// var_proc - variable in process (Relates to something we modified for each process)
	// var_*buffer* - we added some buffer in each process array to Send/Receive boundary values. 

	// Real area without buffer
	int 	  narea	    = npoints * npoints;
	//int     narea_proc = narea / nproc;

	// Area of each process with buffer
	int     narea_proc_with_buffer = (npoints_proc + 4) * (npoints_proc + 4);
	
	double h;

	double elapsed_gpu;
	struct timeval gpu_start, gpu_end;

	// We are calculating co-ordinates (length) for each rank, to make computations easier. 
	// Since we are using 4 processes, the layout is as below 
	// _________
	// 
	//  0    1 
	// 
	//  2    3
 	// 
	// ---------
	// Co-ordinates of rank 0 = (0,0)
	// Co-ordinates of rank 1 = (1,0) . . . etc
	int start_i,start_j;
	switch(rank) {
		case 2 :
			start_i=0;
			start_j=0;
			break;
		case 0 :
			start_i=0;
			start_j=npoints_proc;
			break;
		case 3 :
			start_i=npoints_proc;
			start_j=0;
			break;
		case 1 :
			start_i=npoints_proc;
			start_j=npoints_proc;
			break;
		default :
			printf("Invalid grade \n" );
	}

	double *u, *u_i0, *u_i1,*u_proc,*pebs_proc,*pebs;
	// u will store the final reduced version
	u = (double*)malloc(sizeof(double) * narea);
	pebs = (double*)malloc(sizeof(double) * narea);

	// Variable to store pebs and u values for each process
	pebs_proc = (double*)malloc(sizeof(double) * narea_proc_with_buffer);
	u_proc = (double*)malloc(sizeof(double) * narea_proc_with_buffer);

	// Temporary u variables - for each process 
	u_i0 = (double*)malloc(sizeof(double) * narea_proc_with_buffer);
	u_i1 = (double*)malloc(sizeof(double) * narea_proc_with_buffer);

	// Initialize everything to 0
	memset(u, 0, sizeof(double) * narea);
	memset(pebs, 0, sizeof(double) * narea);
	memset(pebs_proc, 0, sizeof(double) * narea_proc_with_buffer);
	memset(u_proc, 0, sizeof(double) * narea_proc_with_buffer);
	memset(u_i0, 0, sizeof(double) * narea_proc_with_buffer);
	memset(u_i1, 0, sizeof(double) * narea_proc_with_buffer);


	printf("Running %s with (%d x %d) grid, until %f, with %d threads start_i %d start_j %d\n", argv[0], npoints, npoints, end_time, nthreads,start_i, start_j);

	h = (XMAX - XMIN)/npoints;

	if (rank==0) {
		// Compute pebbles at rank 0 ONLY!, as it is a randomized function
		init_pebbles(pebs, npebs, npoints);
	}
	// Communicate pebbles to everyone
	MPI_Bcast(pebs, narea, MPI_DOUBLE, 0, MPI_COMM_WORLD);  


	// Every process splits up their respective pebbles 
	divide(pebs_proc,pebs,npoints_proc,start_i,start_j);

	// Initialize the u values at each process, since they are not randomized
	// We updated init function, the original function is in init_original
	init(u_i0, pebs_proc, npoints_proc);
	init(u_i1, pebs_proc, npoints_proc);

	gettimeofday(&gpu_start, NULL);
	run_gpu(u_proc, u_i0, u_i1, pebs_proc, npoints_proc, h, end_time, nthreads, rank, start_i,start_j);

	gettimeofday(&gpu_end, NULL);
	elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
				gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
	printf("GPU took %f seconds\n", elapsed_gpu);


	// Compute and print the initial pebbles and lake surface values at rank 0
	if(rank==0){
		double *u_initial = (double*)malloc(sizeof(double) * npoints*npoints);
		memset(u_initial, 0, sizeof(double) * npoints*npoints);
		init_original(u_initial,pebs,npoints);
		char filenames[10];
		sprintf(filenames, "lake_i.dat", rank);
		print_heatmap_original(filenames,u_initial, npoints, h);
		free(u_initial);
	}

	// Receive values from other processes and stitch them together
	if(rank==0){
		// u will store the final reduced version
		
		double *u_proc1, *u_proc2, *u_proc3;
		u_proc1 = (double*)malloc(sizeof(double) * (npoints_proc+4) *(npoints_proc+4));
		u_proc2 = (double*)malloc(sizeof(double) * (npoints_proc+4) *(npoints_proc+4));
		u_proc3 = (double*)malloc(sizeof(double) * (npoints_proc+4) *(npoints_proc+4));

		MPI_Request requests[3];
		MPI_Status request_status[3];
		MPI_Irecv(u_proc1, (npoints_proc+4) *(npoints_proc+4), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &requests[0]);
		MPI_Irecv(u_proc2, (npoints_proc+4) *(npoints_proc+4), MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, &requests[1]);
		MPI_Irecv(u_proc3, (npoints_proc+4) *(npoints_proc+4), MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, &requests[2]);
		MPI_Waitall(3,requests,request_status);

		stitch(u,u_proc,u_proc1,u_proc2,u_proc3,npoints_proc);
		print_heatmap_original("lake_f.dat",u, npoints, h);

		free(u_proc1);
		free(u_proc2);
		free(u_proc3);

	// Send u values computed at each process
	} else{
		MPI_Request request[1];
		MPI_Status request_status[1];
		MPI_Isend(u_proc,(npoints_proc+4) *(npoints_proc+4), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&request[0]);
		MPI_Waitall(1,request,request_status);
	}

	// Print heapmap values of each process
	char filename[10];
	sprintf(filename, "lake_f_%d.dat", rank);
	// We updated the print_heatmap function. 
	// The original is in print_heatmap_original
	print_heatmap(filename,u_proc, npoints_proc, h);

	free(u);
	free(u_i0);
	free(u_i1);
	free(pebs);
	free(u_proc);
	MPI_Finalize(); 

	return 0;
}


void init_pebbles(double *p, int pn, int n)
{
	int i, j, k, idx;
	int sz;

	srand( time(NULL) );
	memset(p, 0, sizeof(double) * n * n);

	for( k = 0; k < pn ; k++ )
	{
		i = rand() % (n - 4) + 2;
		j = rand() % (n - 4) + 2;
		sz = rand() % MAX_PSZ;
		idx = j + i * n;
		p[idx] = (double) sz;
	}
}

double f(double p, double t)
{
	return -expf(-TSCALE * t) * p;
}

int tpdt(double *t, double dt, double tf)
{
	if((*t) + dt > tf) return 0;
	(*t) = (*t) + dt;
	return 1;
}

//Function to fill pebble_proc with initial values from pebble(it will fill the matrix excluding the buffer)
void divide(double *pebbles_proc, double *pebbles, int npoints_proc, int start_i,int start_j)
{	
	//npoints_proc => n_without_buffer	
	//idx_pebble is the index in the global pebble array
	//idx_u is the index in the local pebble_proc to which 2 has been added to avoid filling in the buffer
	int i, j, idx_pebble,idx_proc;
	int n_with_buffer = npoints_proc + 4;
	int n_original = npoints_proc*2;
	for(i = 0; i < npoints_proc ; i++)
	{
		for(j = 0; j < npoints_proc ; j++)
		{
			//printf("ij[%d,%d] proc[%d,%d] pebbles[%d,%d]\n",i,j,);
			idx_proc = (j+2) + (i+2) * n_with_buffer;
			idx_pebble = (j+start_j) + (i+start_i) * n_original;
			pebbles_proc[idx_proc] = pebbles[idx_pebble];
		}
	}
}

void init_original(double *u, double *pebbles, int npoints)
{	
	int i, j,idx;
	for(i = 0; i < npoints ; i++)
	{
		for(j = 0; j < npoints ; j++)
		{	
			//Adding 2 to avoid buffer
			idx = (j) + (i) * npoints;
			u[idx] = f(pebbles[idx], 0.0);
		}
	}
}

//Function to fill u0 and u1 with initial values from pebble(it will fill the matrix excluding the buffer)
void init(double *u, double *pebbles_proc, int npoints_proc)
{	
	int i, j,idx;
	int n_with_buffer = npoints_proc+4;
	for(i = 0; i < npoints_proc ; i++)
	{
		for(j = 0; j < npoints_proc ; j++)
		{	
			//Adding 2 to avoid buffer
			idx = (j+2) + (i+2) * n_with_buffer;
			u[idx] = f(pebbles_proc[idx], 0.0);
		}
	}
}
void print_heatmap_original(const char *filename, double *u, int n, double h)
{
	int i, j, idx;
	FILE *fp = fopen(filename, "w");  

	for( i = 0; i < n; i++ )
	{
		for( j = 0; j < n; j++ )
		{
			idx = (j) + (i) * n;
			fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
		}
	}

	fclose(fp);
} 
void stitch(double *u,double *u_proc,double *u_proc1,double *u_proc2,double *u_proc3,int npoints_proc)
{
	int n_with_buffer = npoints_proc+4;
	int npoints = npoints_proc*2;
	int i=0;
	int j=0;
	for(i=0;i<npoints_proc;i++){
		for(j=0;j<npoints_proc;j++){
		int idx0,idx1,idx2,idx3;
	 	int idproc = (i+2)*n_with_buffer +(j+2);
		idx0 = i*npoints +(j+npoints_proc);	
		idx1 = (i+npoints_proc)*npoints +(j+npoints_proc);	
		idx2 = i*npoints +j;	
		idx3 = (i+npoints_proc)*npoints +j;	
		u[idx0]=u_proc[idproc];
		u[idx1]=u_proc1[idproc];
		u[idx2]=u_proc2[idproc];
		u[idx3]=u_proc3[idproc];
		}
	}

}

//n==>nproc_points
void print_heatmap(const char *filename, double *u, int n, double h)
{
	int i, j, idx;
	int n_with_buffer = n+4;
	FILE *fp = fopen(filename, "w");  

	for( i = 0; i < n; i++ )
	{
		for( j = 0; j < n; j++ )
		{
			idx = (j+2) + (i+2) * n_with_buffer;
			fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
		}
	}

	fclose(fp);
} 
