/* Author : zsthampi Zubin S Thampi */

/*
* Compile : mpicc -lm -O3 -o p1 p1.c
* Make    : make -f p1.Makefile
* Run     : prun ./p1
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

/* Main Function */
void main(int argc, char *argv[]) {
  /* Process Info */
  int numproc,rank; 
 
  MPI_Status status;
  /* Variables to capture times */
  double start,end,delta;
  
  /* MPI Initialize */
  MPI_Init(&argc,&argv);
  
  /* Get the number of processes - MPI_Comm_size */
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  /* Get the rank of the process - MPI_Comm_rank  */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Message Info */
  long int size = 16;

  // Maintain a message buffer of the maximum size. 
  // Parts of this is sent as a message, based on the size
  char message[2097152];

  // Fill up message buffer with characters on the root process
  if (rank==0) {
    for (int i=0;i<2097152;i++) {
      message[i] = 'A';
    }
  }

  /* Initialize array to store round trip times */
  double rtt[10];
  double avg_rtt = 0.0;
  double sd_rtt = 0.0;

  // Loop through message sizes - 32 to 2097152
  for (int j=1; j<18 ; j++) {
    size *= 2;
    /* Repeat each message size 10 times */
    for (int i=0;i<10;i++) {
      /* If rank 0, send message to rank 1, receive message from (numproc-1) */
      /* Else if last rank, receive message from (rank-1) and send message to rank 0 */
      /* Else, receive message from (rank-1) and send message to (rank+1) */
      if(rank==0) {
        // Compute time between sending message from process 0, and receiving message at process 0
        start = MPI_Wtime();
        MPI_Send(&message[0],size,MPI_CHAR,rank+1,i,MPI_COMM_WORLD);
        MPI_Recv(&message[0],size,MPI_CHAR,numproc-1,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        end = MPI_Wtime();
        delta = end - start;
        rtt[i] = delta;
        /* printf("Size %ld : Iteration %d/10 : Round Trip Time = %2.12lf \n",size,i+1,delta); */
      } else if(rank==(numproc-1)) {
        MPI_Recv(&message[0],size,MPI_CHAR,rank-1,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        /* printf("Process %d received message from %d with tag %d \n",rank,status.MPI_SOURCE,status.MPI_TAG); */
        MPI_Send(&message[0],size,MPI_CHAR,0,i,MPI_COMM_WORLD);
      } else {
        MPI_Recv(&message[0],size,MPI_CHAR,rank-1,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        /* printf("Process %d received message from %d with tag %d \n",rank,status.MPI_SOURCE,status.MPI_TAG); */
        MPI_Send(&message[0],size,MPI_CHAR,rank+1,i,MPI_COMM_WORLD);
      }
    }
    
    /* Compute average round trip time */
    avg_rtt = 0.0;
    sd_rtt = 0.0;
    if (rank==0) {
      for (int i=1;i<10;i++) {
        avg_rtt += rtt[i];
      }
      avg_rtt /= (double)9;
      for (int i=1;i<10;i++) {
        sd_rtt += pow(rtt[i] - avg_rtt, 2);
      }
      sd_rtt = sqrt(sd_rtt/(double)9);
      printf("%ld %2.12lf %2.12lf\n",size,avg_rtt,sd_rtt);
    }
  }

  /* MPI Finalize */
  MPI_Finalize();
}

