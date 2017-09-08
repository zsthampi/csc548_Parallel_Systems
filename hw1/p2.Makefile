# zsthampi Zubin S Thampi
# sgarg7 Shaurya Garg
# kjadhav Karan Jadhav

CC=mpicc
CFLAGS=-lm

p2_mpi: p2_mpi.o
	$(CC) -O3 -o p2_mpi p2_mpi.c $(CFLAGS)
