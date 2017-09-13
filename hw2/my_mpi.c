// Author : zsthampi Zubin S Thampi

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <time.h>
#include <sys/time.h>
#include "my_mpi.h"

int sendsockfd, readsockfd, newreadsockfd, n;
struct sockaddr_in send_serv_addr, recv_serv_addr, cli_addr;
socklen_t clilen;
struct hostent *server;
struct timeval start,end;

int nproc, rank;

int generate_random() {
	// I'm just using a constant port number for all the processes. 
	// Since the probability that a port would be occupied is the same if I generate it randomly
	// You can update the code here, to use a different port number
	return 29717;
}

double time_diff(struct timeval x , struct timeval y)
{
    double x_ms , y_ms , diff;
     
    x_ms = (double)x.tv_sec*1000000 + (double)x.tv_usec;
    y_ms = (double)y.tv_sec*1000000 + (double)y.tv_usec;
     
    diff = (double)y_ms - (double)x_ms;
     
    return diff/1000000;
}

int MPI_Init(int *argc, char *argv[]) {
	nproc = atoi(argv[1]);
	rank = atoi(argv[2]);

	gettimeofday(&start , NULL);

	int dest;
	int portno = generate_random();

	if (rank==(nproc-1)) {
		dest = 0;
	} else {
		dest = rank + 1;
	}

	// Create socket, and bind
	readsockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (readsockfd<0) {
		printf("ERROR opening socket \n");
		return 1;
	}

	recv_serv_addr.sin_family = AF_INET;
	recv_serv_addr.sin_addr.s_addr = INADDR_ANY;
	recv_serv_addr.sin_port = htons(portno);
	if (bind(readsockfd, (struct sockaddr *) &recv_serv_addr, sizeof(recv_serv_addr)) < 0) {
		printf("ERROR on binding \n");
		return 1;
	}
	listen(readsockfd,5);

	// Write to file to sync up
	FILE *fptr;
	fptr = fopen("sync", "a");
	fprintf(fptr,"A");
	fclose(fptr);

	// Read from file to ensure everyone is synced up
	int n_active = 0;
	char verify_buffer[nproc];
	while (n_active<nproc) {
		fptr = fopen("sync", "r");
		fscanf(fptr, "%s", verify_buffer);
		n_active = strlen(verify_buffer);
	}

	// Sleep to sync up with processes
	// sleep(5);
	
	// Connect to the bind
	sendsockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sendsockfd<0) {
		printf("ERROR opening socket \n");
		return 1;
	}
	server = gethostbyname(get_host_name(dest));
	if (server==NULL) {
		printf("ERROR : server is null \n");
	}

	bzero((char *) &send_serv_addr, sizeof(send_serv_addr));
	send_serv_addr.sin_family = AF_INET;
	bcopy((char *)server->h_addr, (char *)&send_serv_addr.sin_addr.s_addr, server->h_length);
	send_serv_addr.sin_port = htons(portno);
	if (connect(sendsockfd,(struct sockaddr *) &send_serv_addr,sizeof(send_serv_addr))<0) {
		printf("ERROR on connect \n");
		return 1;
	}
	
	// Accept the connection
	clilen = sizeof(cli_addr);
	newreadsockfd = accept(readsockfd, (struct sockaddr *) &cli_addr, &clilen);
	if (newreadsockfd<0) {
		printf("ERROR accepting connection \n");
		return 1;
	}
	return 0;
} 

void MPI_Finalize() {
	MPI_Barrier();

	close(sendsockfd);
	close(newreadsockfd);
	close(readsockfd);
	return 0;
}

void MPI_Barrier() {
	// Barrier uses 2 round trip messages 
	// 1 - to handshake with everyone 
	// 2 - to notify everyone that the handshake is complete

	char handshake = 'A';
	// rank 0
	// starts a rountrip message for handshake
	// receives the handshake from (nproc-1) - every process is at the same step now and is synced up 
	// starts another rountrip message to notify other processes
	if (rank==0) {
		n = write(sendsockfd,&handshake,1);
		n = read(newreadsockfd,&handshake,1);
		n = write(sendsockfd,&handshake,1);
	} 
	// rank (n-1) 
	// receive message from previous process for the handshake
	// sends message to process 0 to complete the handshake
	// receives message from previous process for the notification
	else if (rank==(nproc-1)) {
		n = read(newreadsockfd,&handshake,1);
		n = write(sendsockfd,&handshake,1);
		n = read(newreadsockfd,&handshake,1);
	} 
	// all other rank
	// receives message from previous process for the handshake
	// sends message to next process for the handshake
	// receives message from previous process for the notification
	// sends message to next process for the notification 
	else {
		n = read(newreadsockfd,&handshake,1);
		n = write(sendsockfd,&handshake,1);
		n = read(newreadsockfd,&handshake,1);
		n = write(sendsockfd,&handshake,1);
	}

	return 0;
}

double MPI_Wtime() {
	gettimeofday(&end , NULL);
	return time_diff(start,end);
}

void MPI_Comm_size(int channel,int *numproc) {
	*numproc = nproc;
}

void MPI_Comm_rank(int channel,int *rank_holder) {
	*rank_holder = rank;
}

void MPI_Send(void *message,int size,int type,int dest,int tag,int channel) {
	n = write(sendsockfd,message,size);
	
	char buffer[18];
	bzero(buffer,18);
	n = read(sendsockfd,buffer,18);
}

void MPI_Recv(void *buffer,int size,int type,int source,int tag,int channel,struct MPI_Status *status) {
	n = read(newreadsockfd,buffer,size);
	n = write(newreadsockfd,"I got your message",18);
}

char* get_host_name(int rank) {
	FILE *fp;
	fp = fopen("nodefile.txt", "r");

	char line[256];
	for (int i=0; i<=rank; i++) {
		fgets(line, 256, fp);
	}

	fclose(fp);

	// Remove the newline at the end
	if (line[strlen(line)-1]=='\n')
	line[strlen(line)-1] = '\0';

	char * line_buffer;
	line_buffer =(char*)malloc(sizeof(char) * strlen(line));
	strcpy(line_buffer, line);

	return line_buffer;
}


