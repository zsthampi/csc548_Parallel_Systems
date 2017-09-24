#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Initialize an array of count to store the number of MPI calls 
int *count;
int nproc;
int rank;

int MPI_Init(int *argc, char ***argv) {
	PMPI_Init(argc, argv);

	// Get the number of processes and the rank
	PMPI_Comm_size(MPI_COMM_WORLD, &nproc);
	PMPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Assign memory to count array, and initialize all values to 0
	count = (int *) malloc(sizeof(int) * nproc);
	for (int i=0; i<nproc; i++) {
		count[i] = 0;
	}
}

int MPI_Send(const void *buf, int size, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
	count[dest] += 1;
	PMPI_Send(buf, size, datatype, dest, tag, comm); 
}

int MPI_Finalize() {
	if (rank==0) {
		FILE *f;
		f = fopen("matrix.data","w");

		fprintf(f, "%d", rank);
		for (int i=0; i<nproc; i++) {
			fprintf(f, " %d", count[i]);
		}
		fprintf(f, "\n");

		// Receive corresponding count values from other ranks
		for (int j=1; j<nproc; j++) {

			PMPI_Recv(&count[0], nproc, MPI_INT, j, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			fprintf(f, "%d", j);
			for (int i=0; i<nproc; i++) {
				fprintf(f, " %d",count[i]);
			}
			fprintf(f, "\n");
		}

		fclose(f);
	} else {
		// Send the count values to rank 0
		PMPI_Send(&count[0], nproc, MPI_INT, 0, rank, MPI_COMM_WORLD);
	}

	PMPI_Finalize();
}
