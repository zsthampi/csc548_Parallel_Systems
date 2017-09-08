#include <stdio.h>
#include <stdlib.h>

void main(int argc, char *argv[]) {
	if (argc<3) {
		printf("Usage : ./hello_world <nproc> <rank>");
	}
	int nproc = atoi(argv[1]);
	int rank = atoi(argv[2]);
	printf("Hello World! %d / %d \n",rank,nproc);
}

