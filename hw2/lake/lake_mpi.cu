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

void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(const char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);

extern void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads);

int main(int argc, char *argv[])
{
  setbuf(stdout, NULL);

  // Maybe change to global variables if extern does not work
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
  // The number of points inside each process
  int     npoints_proc = npoints / sqrt(nproc);
  int     npebs     = atoi(argv[2]);
  double  end_time  = (double)atof(argv[3]);
  int     nthreads  = atoi(argv[4]);
  int 	  narea	    = npoints * npoints;
  int     narea_proc = narea / nproc;
  int     narea_proc_with_buffer = (npoints_proc + 4) * (npoints_proc + 4);

  double h;

  double elapsed_cpu, elapsed_gpu;
  struct timeval cpu_start, cpu_end, gpu_start, gpu_end;
  

  double *u, *u_i0, *u_i1;
  double *pebs;
  u = (double*)malloc(sizeof(double) * narea);
  u_i0 = (double*)malloc(sizeof(double) * narea);
  u_i1 = (double*)malloc(sizeof(double) * narea);
  pebs = (double*)malloc(sizeof(double) * narea);

  // Commenting out CPU code in the whole code
  // double *u_cpu;
  // u_cpu = (double*)malloc(sizeof(double) * narea_with_buffer);

  double *u_gpu, *pebs_gpu;
  double *u_j0, *u_j1;
  u_j0 = (double*)malloc(sizeof(double) * narea_proc_with_buffer);
  u_j1 = (double*)malloc(sizeof(double) * narea_proc_with_buffer);
  u_gpu = (double*)malloc(sizeof(double) * narea_proc_with_buffer);
  pebs_gpu = (double*)malloc(sizeof(double) * narea_proc_with_buffer);

  printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints, end_time, nthreads);

  h = (XMAX - XMIN)/npoints;

  if (rank==0) {
    // Compute and communicate pebbles as it is a randomized function
    init_pebbles(pebs, npebs, npoints);
  }
  // !!!TODO!!! We can change this to divide the pebbles at rank 0, and broadcast respective arrays to each process
  MPI_Bcast(pebs, narea, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
  
  // !!!TODO!!! Compute u values at each process, as they are fixed values
  // !!!TODO!!! This will avoid the overhead in MPI Communication
  init(u_i0, pebs, npoints);
  init(u_i1, pebs, npoints);

  // !!!TODO!!! REMOVE the bigger U tables
  int idx;
  int count = 0;
  for (int i=0; i<npoints_proc+4; i++) {
    for (int j=0; j<npoints_proc+4; j++) {
      idx = j + i * (npoints_proc+4);
      if (i<2 || j<2 || i>npoints_proc+1 || j>npoints_proc+1) {
        u_j0[idx] = 0.;
        u_j1[idx] = 0.;
        pebs_gpu[idx] = 0.;
      } else {
        u_j0[idx] = u_i0[count];
        u_j1[idx] = u_i1[count];
        pebs_gpu[idx] = pebs[count];
        count++;
      }
    }
  }
  	
  print_heatmap("lake_i.dat", u_i0, npoints, h);

  // gettimeofday(&cpu_start, NULL);
  // run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time);
  // gettimeofday(&cpu_end, NULL);

  // elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
  //                 cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
  // printf("CPU took %f seconds\n", elapsed_cpu);

  gettimeofday(&gpu_start, NULL);
  run_gpu(u_gpu, u_j0, u_j1, pebs_gpu, npoints_proc, h, end_time, nthreads);

  // !!!TODO!!! MPI_Reduce using rank0 and u_gpu to u

  gettimeofday(&gpu_end, NULL);
  elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                  gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
  printf("GPU took %f seconds\n", elapsed_gpu);


  // // DEBUG CODE 
  
  // // printf("\n OUTPUT \n ========= \n");
  // // for (int i=0; i<narea; i++) {
  // //   printf("%2.4lf : %2.4lf \n",u_cpu[i],u_gpu[i]);
  // // }

  // // print_heatmap("lake_f.dat", u_cpu, npoints, h);
  print_heatmap("lake_f .dat", u, npoints, h);

  free(u_i0);
  free(u_i1);
  free(u_j0);
  free(u_j1);
  free(pebs);
  free(pebs_gpu);
  // free(u_cpu);
  free(u_gpu);

  MPI_Finalize(); 

  return 1;
}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
{
  double *un, *uc, *uo;
  double t, dt;

  un = (double*)malloc(sizeof(double) * n * n);
  uc = (double*)malloc(sizeof(double) * n * n);
  uo = (double*)malloc(sizeof(double) * n * n);

  memcpy(uo, u0, sizeof(double) * n * n);
  memcpy(uc, u1, sizeof(double) * n * n);

  t = 0.;
  dt = h / 2.;

  while(1)
  {
    evolve(un, uc, uo, pebbles, n, h, dt, t);

    memcpy(uo, uc, sizeof(double) * n * n);
    memcpy(uc, un, sizeof(double) * n * n);

    if(!tpdt(&t,dt,end_time)) break;
  }
  
  memcpy(u, un, sizeof(double) * n * n);
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

void init(double *u, double *pebbles, int n)
{
  int i, j, idx;

  for(i = 0; i < n ; i++)
  {
    for(j = 0; j < n ; j++)
    {
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}

void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i * n;

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
                  0.125*(uc[idx-2] + uc[idx+2] + uc[idx+2*n] + uc[idx-2*n]) - 6 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
    }
  }
}

void print_heatmap(const char *filename, double *u, int n, double h)
{
  int i, j, idx;

  FILE *fp = fopen(filename, "w");  

  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      idx = j + i * n;
      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }
  
  fclose(fp);
} 