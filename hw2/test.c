#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "my_mpi.h"


void main(int argc, char* argv[]) {
	int numproc,rank; 

	MPI_Init(argc,argv);

	/* Get the number of processes - MPI_Comm_size */
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);

	/* Get the rank of the process - MPI_Comm_rank  */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("%d / %d",rank,numproc);
}