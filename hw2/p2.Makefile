# Author : zsthampi Zubin S Thampi

CC=nvcc

p2: p2.cu
	$(CC) p2.cu -o p2 -O3 -lm -Wno-deprecated-gpu-targets