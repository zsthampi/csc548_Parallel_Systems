/* Program to compute Pi using the Leibniz formula */

#include <stdio.h>
#include <math.h>
#include "mytime.h"

double Leibniz(int n) {
  double sum;
  int k;

  sum = 0.0;
  for (k=0; k<n; k++) {
	sum += pow(-1,k) / (2*k + 1);
  }

  return sum*4;
}

int main(int argc, char *argv[]) {
  int n;
  double PI25DT = 3.141592653589793238462643;
  double pi;
  struct timeval startwtime, endwtime, diffwtime;
  
  while (1) {
    printf("Enter the number of intervals: (0 quits) ");fflush(stdout);
    scanf("%d",&n);

    gettimeofday(&startwtime, NULL);
    if (n == 0)
      break;
    pi = Leibniz(n);
    
    gettimeofday(&endwtime, NULL);
    MINUS_UTIME(diffwtime, endwtime, startwtime);
    printf("pi is approximately %.16f, Error is %.16f\n",
	   pi, fabs(pi - PI25DT));
    printf("wall clock time = %d.%06d\n",
	   diffwtime.tv_sec, diffwtime.tv_usec);
  }

  return 0;
}
