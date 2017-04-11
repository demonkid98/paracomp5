#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1024

void print_array(int *M, int size) {
  register int i, j;
  printf("-----\n");
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      printf("%d ", M[i * size + j]);
    }
    printf("\n");
  }
}

int main(int argc, char* argv[]) {
  clock_t st, ed;
  int i, j, l;
  static int A[N * N];
  static int B[N * N];
  static int C[N * N];

  st = clock();
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      // A[i * N + j] = i * N + j;
      // B[i * N + j] = i * N + j;
      A[i * N + j] = i;
      B[i * N + j] = j;
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      // C[i][j]
      for (l = 0; l < N; l++) {
        C[i * N + j] = C[i * N + j] + A[i * N + l] * B[l * N + j];
      }
    }
  }

  // show result
  // print_array(C, N);
  ed = clock();
  printf(">> Exec time: %fs\n", ((double) (ed - st)) / 1000000);
}
