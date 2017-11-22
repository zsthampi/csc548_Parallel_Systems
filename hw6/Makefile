all:
	mpicc -o TFIDF TFIDF.c -lm -O3
	mpicc -o TFIDF_serial TFIDF_serial.c -lm -O3

clean:
	rm -f TFIDF
	rm -f TFIDF_serial
