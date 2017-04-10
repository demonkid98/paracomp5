#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>

#define NDIMS 2
#define N 6

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
  int rank, nrank, uprank, downrank;
  int size, bsize;
  int i, j, k, l;
  int nnodes = 0;
  int dims[NDIMS];
  int periods[NDIMS] = {1, 1};
  int coords[NDIMS];


  MPI_Request send_req;
  MPI_Request recv_req;
  MPI_Status send_status;
  MPI_Status recv_status;

  MPI_Datatype sub_matrix;

  for (i = 0; i < NDIMS; i++) {
    dims[i] = 0;
  }
  MPI_Comm grid_comm, row_comm, col_comm;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Dims_create(size, NDIMS, dims);
  MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, 0, &grid_comm);
  MPI_Comm_rank(grid_comm, &nrank);
  MPI_Cart_coords(grid_comm, nrank, NDIMS, coords);
  MPI_Cart_shift(grid_comm, 0, 1, &uprank, &downrank);

  // printf("[%d] up %d down %d\n", rank, uprank, downrank);
  int A[N * N];
  int B[N * N];
  int C[N * N];

  int sendcounts[size];
  int displ[size];

  bsize = N / dims[0];

  MPI_Type_vector(bsize, bsize, N, MPI_INT, &sub_matrix);
  MPI_Type_commit(&sub_matrix);


  for (i = 0; i < size; i++) {
    sendcounts[i] = 1;
    int i1 = i / dims[0];
    int i2 = i % dims[0];
    // displ[i] = i1 * bsize * N + i2 * bsize;
    displ[i] = 1;
    // printf("[%d] displ %d\n", i, displ[i]);
  }

  int Aloc[bsize * bsize];
  int Bloc[bsize * bsize];
  int Cloc[bsize * bsize];
  int Abuf[bsize * bsize];
  int Brecv[bsize * bsize];

  memset(Cloc, 0, sizeof(Cloc));

  int row_remain_dims[NDIMS] = {0, 1};
  MPI_Cart_sub(grid_comm, row_remain_dims, &row_comm);
  int col_remain_dims[NDIMS] = {1, 0};
  MPI_Cart_sub(grid_comm, col_remain_dims, &col_comm);

  if (rank == 0) {
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        A[i * N + j] = i * N + j;
        B[i * N + j] = i * N + j;
      }
    }
  }

  // TODO not working yet!
  MPI_Scatterv(&A[0], sendcounts, displ, sub_matrix, Aloc, bsize * bsize, MPI_INT, 0, MPI_COMM_WORLD);
  // if (rank == 1)
    print_array(Aloc, bsize);


  MPI_Finalize();
}
