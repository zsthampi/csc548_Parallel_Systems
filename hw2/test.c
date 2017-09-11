#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "my_mpi.h"


void main(int argc, char* argv[]) {
	int numproc,rank; 

	printf("Init \n");
	MPI_Init(argc,argv);

	printf("Comm size \n");
	/* Get the number of processes - MPI_Comm_size */
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);

	printf("Comm rank \n");
	/* Get the rank of the process - MPI_Comm_rank  */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char message = 'A';
	struct MPI_Status status;

	if (rank==0) {
		printf("MPI Send \n");
		MPI_Send(&message,1,MPI_CHAR,1,0,MPI_COMM_WORLD);
	} else {
		printf("MPI Recv \n");
		MPI_Recv(&message,1,MPI_CHAR,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
	}

	printf("%d / %d \n",rank,numproc);
}
