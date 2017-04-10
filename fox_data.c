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
  MPI_Datatype sub_matrix2;

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
  // resize type to allow data interleaving (i.e. in same row group)
  MPI_Type_create_resized(sub_matrix, 0, sizeof(int), &sub_matrix2);
  MPI_Type_commit(&sub_matrix2);

  for (i = 0; i < size; i++) {
    sendcounts[i] = 1;
    int i1 = i / dims[0];
    int i2 = i % dims[0];
    displ[i] = i1 * bsize * N + i2 * bsize;
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

  MPI_Scatterv(A, sendcounts, displ, sub_matrix2, Aloc, bsize * bsize, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(B, sendcounts, displ, sub_matrix2, Bloc, bsize * bsize, MPI_INT, 0, MPI_COMM_WORLD);
  // if (rank == 3) print_array(Bloc, bsize);
 
  for (k = 0; k < dims[0]; k++) {
    // start shifting B async
    // data is sent to a temporary buffer and copied back later
    MPI_Isend(Bloc, bsize * bsize, MPI_INT, uprank, k * nrank, grid_comm, &send_req);
    MPI_Irecv(Brecv, bsize * bsize, MPI_INT, downrank, k * downrank, grid_comm, &recv_req);

    // bcast A row-wise
    // each row
    for (i = 0; i < dims[0]; i++) {
      if (i != coords[0]) {
        continue;
      }
      int row_rank;
      MPI_Comm_rank(row_comm, &row_rank);

      // root of bcast row-wise: [i][(i + k) mod dims[0]]
      int root_col = (i + k) % dims[0];
      int root_coords[NDIMS] = {i, (i + k) % dims[0]};
      int root;
      MPI_Cart_rank(grid_comm, root_coords, &root);

      if (rank == root) {
        memcpy(Abuf, Aloc, sizeof(int) * bsize * bsize);
      }

      MPI_Bcast(Abuf, bsize * bsize, MPI_INT, root_col, row_comm);
    }

    // local compute
    for (i = 0; i < bsize; i++) {
      for (j = 0; j < bsize; j++) {
        // Cloc[i][j]
        for (l = 0; l < bsize; l++) {
          Cloc[i * bsize + j] = Cloc[i * bsize + j] + Abuf[i * bsize + l] * Bloc[l * bsize + j];
        }
      }
    }

    // wait for shift B
    MPI_Wait(&send_req, &send_status);
    MPI_Wait(&recv_req, &recv_status);

    // barrier to ensure all data are transferred to the target buffer completely
    // before copying to B
    MPI_Barrier(grid_comm);
    memcpy(Bloc, Brecv, sizeof(int) * bsize * bsize);
  }

  MPI_Gatherv(Cloc, bsize * bsize, MPI_INT, C, sendcounts, displ, sub_matrix2, 0, MPI_COMM_WORLD);

  // show result
  if (rank == 0) print_array(C, N);
  MPI_Finalize();
}
