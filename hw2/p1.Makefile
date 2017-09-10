# Author : zsthampi Zubin S Thampi

CC=gcc

all: my_mpi test

my_mpi: my_mpi.c
	$(CC) -o my_mpi my_mpi.c -c

test: test.c my_mpi.c
	$(CC) -o test test.c my_mpi.c
