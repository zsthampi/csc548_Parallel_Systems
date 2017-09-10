#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include "my_mpi.h"

int sockfd, newsockfd, n;
int portno = 51718;
struct sockaddr_in serv_addr, cli_addr;
socklen_t clilen;
struct hostent *server;

int nproc, rank;

int MPI_Init(int *argc, char *argv[]) {
	nproc = atoi(argv[1]);
	rank = atoi(argv[2]);
	return 0;
} 

void MPI_Finalize() {
	return 0;
}

void MPI_Barrier() {
	return 0;
}

double MPI_Wtime() {
	return 0.0;
}

void MPI_Comm_size(int channel,int *numproc) {
	*numproc = nproc;
}

void MPI_Comm_rank(int channel,int *rank_holder) {
	*rank_holder = rank;
}

void MPI_Send(void *message,int size,int type,int dest,int tag,int channel) {
	printf("Send Begin \n");
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	printf("sockfd : %d \n",sockfd);
	printf("Hostname : %s \n",get_host_name(dest));
	server = gethostbyname(get_host_name(dest));
	if (server==NULL) {
		printf("ERROR : server is null \n");
	} else {
		printf("Server initialized \n");
	}
	bzero((char *) &serv_addr, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr, server->h_length);
	serv_addr.sin_port = htons(portno);
	printf("Connect : %d \n",connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)));
	n = write(sockfd,message,size);
	printf("Write : %d \n",n);
	
	char buffer[18];
	bzero(buffer,18);
	n = read(sockfd,buffer,18);

	printf("Read : %d \n",n);
	close(sockfd);
	printf("Send End \n");
}

void MPI_Recv(void *buffer,int size,int type,int source,int tag,int channel,struct MPI_Status *status) {
	printf("Recv BEGIN \n");
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	printf("sockfd : %d \n",sockfd);
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = INADDR_ANY;
	serv_addr.sin_port = htons(portno);
	printf("Bind : %d \n", bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)));
	listen(sockfd,5);
	clilen = sizeof(cli_addr);
	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	printf("newsockfd : %d \n",newsockfd);
	n = read(newsockfd,buffer,size);
	printf("Read : %d \n",n);
	n = write(newsockfd,"I got your message",18);
	printf("Write : %d \n",n);

	close(newsockfd);
	close(sockfd);
	printf("Recv END \n");
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
	line[strlen(line)-1] = 0;

	char * line_buffer;
	line_buffer =(char*)malloc(sizeof(char) * strlen(line));
	strcpy(line_buffer, line);

	return line_buffer;
}


