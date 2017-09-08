# Author : zsthampi Zubin S Thampi

CC=mpicc
CFLAGS=-lm

p1: p1.o
	$(CC) -O3 -o p1 p1.c $(CFLAGS)
