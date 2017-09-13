// Author : zsthampi Zubin S Thampi

// Define MPI_COMM_WORLD, MPI_ANY_TAG, MPI_CHAR
#define MPI_COMM_WORLD 0
#define MPI_ANY_TAG 0
#define MPI_CHAR 0

struct MPI_Status {
	int MPI_SOURCE;
	int MPI_TAG;
};

int generate_random();
double time_diff(struct timeval x , struct timeval y);
int MPI_Init(int *argc, char *argv[]);
void MPI_Finalize();
void MPI_Barrier();
void MPI_Comm_size(int channel,int *numproc);
void MPI_Comm_rank(int channel,int *rank);
void MPI_Send(void *message,int size,int type,int dest,int tag,int channel);
void MPI_Recv(void *buffer,int size,int type,int source,int tag,int channel,struct MPI_Status *status);
char* get_host_name(int rank);
double MPI_Wtime();
