# Author : zsthampi Zubin S Thampi

CC=gcc

all: my_mpi my_rtt

my_mpi: my_mpi.c
	$(CC) -o my_mpi my_mpi.c -c

my_rtt: my_rtt.c my_mpi.c
	$(CC) -o my_rtt my_rtt.c my_mpi.c -lm
