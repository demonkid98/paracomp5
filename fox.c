#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>

#define NDIMS 2
#define N 9

typedef int matrix[N][N];

void print_array(matrix M, int size) {
  register int i, j;
  printf("-----\n");
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      printf("%d ", M[i][j]);
    }
    printf("\n");
  }
}

int main(int argc, char* argv[]) {
  int rank, nrank, uprank, downrank;
  int size, bsize;
  int i, j, k, l;
  int nnodes = 0;
  int dims[NDIMS];
  int periods[NDIMS] = {1, 1};
  int coords[NDIMS];

  matrix A, B, C;
  matrix myA, Bbuffer;

  MPI_Request req;
  MPI_Status status;

  for (i = 0; i < NDIMS; i++) {
    dims[i] = 0;
  }
  MPI_Comm new_comm, row_comm, col_comm;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Dims_create(size, NDIMS, dims);
  MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, 0, &new_comm);
  MPI_Comm_rank(new_comm, &nrank);
  MPI_Cart_coords(new_comm, nrank, NDIMS, coords);
  MPI_Cart_shift(new_comm, 0, 1, &uprank, &downrank);

  // printf("[%d] up %d down %d\n", rank, uprank, downrank);

  bsize = N / dims[0];

  // assume blocks are already in each process
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i][j] = 0;

      if (coords[0] == i / bsize && coords[1] == j / bsize) {
        // if (i == j) {
        //   A[i][j] = i + 1;
        //   B[i][j] = N - i;  
        // } else {
        //   A[i][j] = 0;
        //   B[i][j] = 0;
        // }
        A[i][j] = rank + 1;
        // B[i][j] = i + 1;
        B[i][j] = 1;
      } else {
        A[i][j] = -1;
        B[i][j] = -1;  
      }
    }
  }

  memcpy(myA, A, sizeof(int) * N * N);
  memcpy(Bbuffer, B, sizeof(int) * N * N);  

  int row_remain_dims[NDIMS] = {1, 0};
  MPI_Cart_sub(new_comm, row_remain_dims, &row_comm);
  int col_remain_dims[NDIMS] = {0, 1};
  MPI_Cart_sub(new_comm, col_remain_dims, &col_comm);

  int i0, j0;
  i0 = coords[0] * bsize;
  j0 = coords[1] * bsize;
  // printf("[%d][%d] %d %d\n", coords[0], coords[1], i0, j0);

  for (k = 0; k < dims[0]; k++) {
    // bcast k-th diag    
    for (i = 0; i < N; i++) {
      // FIXME only broadcast on row
      int root = i / bsize;
      root = root * dims[1] + ((root + k) % dims[1]);
      int j0 = ((i / bsize + k) % dims[1]) * bsize;

      MPI_Bcast(&A[i][j0], bsize, MPI_INT, root, MPI_COMM_WORLD);
      for (j = 0; j < dims[1]; j++) {
        memcpy(&A[i][j * bsize], &A[i][j0], sizeof(int) * bsize);
      }
    }

    // local compute
    for (i = i0; i < i0 + bsize; i++) {
      for (j = j0; j < j0 + bsize; j++) {
        // C[i][j]
        for (l = 0; l < bsize; l++) {
          C[i][j] = C[i][j] + A[i][j0 + l] * B[i0 + l][j];
        }
      }
    }

    // restore A
    memcpy(A, myA, sizeof(int) * N * N);

    // shift B
    for (i = i0; i < i0 + bsize; i++) {
      memcpy(&Bbuffer[i][j0], &B[i][j0], sizeof(int) * bsize);
      MPI_Irecv(&Bbuffer[i][j0], bsize, MPI_INT, downrank, k * downrank, new_comm, &req);
      MPI_Send(&B[i][j0], bsize, MPI_INT, uprank, k * nrank, new_comm);
      MPI_Wait(&req, &status);
    }

    // barrier to ensure all data are transferred to the target buffer completely
    // before copying to B
    MPI_Barrier(new_comm);
    for (i = i0; i < i0 + bsize; i++) {
      memcpy(&B[i][j0], &Bbuffer[i][j0], sizeof(int) * bsize);
    }
  }

  for (i = 0; i < size; i++) {
    MPI_Barrier(new_comm);
    if (rank == i) {
      printf("[%d]-- Result --\n", i);
      print_array(C, N);
    }
    MPI_Barrier(new_comm);
  }

  // no gather at the end


  MPI_Finalize();
}
