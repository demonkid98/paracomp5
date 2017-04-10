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
  int rank, nrank,
      uprank, downrank,
      leftrank, rightrank,
      row_skew_rank, irow_skew_rank,
      col_skew_rank, icol_skew_rank;
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
  MPI_Cart_shift(grid_comm, 1, 1, &leftrank, &rightrank); // my neighbor in row dimension
  MPI_Cart_shift(grid_comm, 0, 1, &uprank, &downrank); // my neighbor in col dimension
  MPI_Cart_shift(grid_comm, 1, coords[0], &row_skew_rank, &irow_skew_rank); // row i -- skew A to the left i cols
  MPI_Cart_shift(grid_comm, 0, coords[1], &col_skew_rank, &icol_skew_rank); // col j -- skew B to the top j cols

  // printf("[%d] up %d down %d\n", rank, uprank, downrank);

  bsize = N / dims[0];

  int Aloc[bsize * bsize];
  int Bloc[bsize * bsize];
  int Cloc[bsize * bsize];
  int Arecv[bsize * bsize];
  int Brecv[bsize * bsize];

  for (i = 0; i < bsize; i++) {
    for (j = 0; j < bsize; j++) {
      Aloc[i * bsize + j] = rank;
      Bloc[i * bsize + j] = rank;
    }
  }
  memset(Cloc, 0, sizeof(Cloc));

  int row_remain_dims[NDIMS] = {0, 1};
  MPI_Cart_sub(grid_comm, row_remain_dims, &row_comm);
  int col_remain_dims[NDIMS] = {1, 0};
  MPI_Cart_sub(grid_comm, col_remain_dims, &col_comm);

  // preskew A (except row 0)
  if (coords[0] != 0) {
    MPI_Isend(Aloc, bsize * bsize, MPI_INT, row_skew_rank, nrank, grid_comm, &send_req);
    MPI_Irecv(Arecv, bsize * bsize, MPI_INT, irow_skew_rank, irow_skew_rank, grid_comm, &recv_req);
    MPI_Wait(&send_req, &send_status);
    MPI_Wait(&recv_req, &recv_status);
  }

  // preskew B (except col 0)
  if (coords[1] != 0) {
    MPI_Isend(Bloc, bsize * bsize, MPI_INT, col_skew_rank, nrank, grid_comm, &send_req);
    MPI_Irecv(Brecv, bsize * bsize, MPI_INT, icol_skew_rank, icol_skew_rank, grid_comm, &recv_req);
    MPI_Wait(&send_req, &send_status);
    MPI_Wait(&recv_req, &recv_status);
  }

  MPI_Barrier(grid_comm);
  if (coords[0] != 0) {
    memcpy(Aloc, Arecv, sizeof(int) * bsize * bsize);
  }
  if (coords[1] != 0) {
    memcpy(Bloc, Brecv, sizeof(int) * bsize * bsize);
  }


  for (k = 0; k < dims[0]; k++) {
    // shift A left (async) -- copy at the end
    MPI_Isend(Aloc, bsize * bsize, MPI_INT, leftrank, nrank, grid_comm, &send_req);
    MPI_Irecv(Arecv, bsize * bsize, MPI_INT, rightrank, rightrank, grid_comm, &recv_req);

    // shift B up (async) -- copy at the end
    MPI_Isend(Bloc, bsize * bsize, MPI_INT, uprank, nrank, grid_comm, &send_req);
    MPI_Irecv(Brecv, bsize * bsize, MPI_INT, downrank, downrank, grid_comm, &recv_req);

    // local compute
    for (i = 0; i < bsize; i++) {
      for (j = 0; j < bsize; j++) {
        // Cloc[i][j]
        for (l = 0; l < bsize; l++) {
          Cloc[i * bsize + j] = Cloc[i * bsize + j] + Aloc[i * bsize + l] * Bloc[l * bsize + j];
        }
      }
    }

    // copy shift result from buffer
    MPI_Barrier(grid_comm);
    memcpy(Aloc, Arecv, sizeof(int) * bsize * bsize);
    memcpy(Bloc, Brecv, sizeof(int) * bsize * bsize);
  }

  // postskew A (except row 0)
  if (coords[0] != 0) {
    MPI_Isend(Aloc, bsize * bsize, MPI_INT, irow_skew_rank, nrank, grid_comm, &send_req);
    MPI_Irecv(Arecv, bsize * bsize, MPI_INT, row_skew_rank, row_skew_rank, grid_comm, &recv_req);
    MPI_Wait(&send_req, &send_status);
    MPI_Wait(&recv_req, &recv_status);
  }

  // postskew B (except col 0)
  if (coords[1] != 0) {
    MPI_Isend(Bloc, bsize * bsize, MPI_INT, icol_skew_rank, nrank, grid_comm, &send_req);
    MPI_Irecv(Brecv, bsize * bsize, MPI_INT, col_skew_rank, col_skew_rank, grid_comm, &recv_req);
    MPI_Wait(&send_req, &send_status);
    MPI_Wait(&recv_req, &recv_status);
  }

  MPI_Barrier(grid_comm);
  if (coords[0] != 0) {
    memcpy(Aloc, Arecv, sizeof(int) * bsize * bsize);
  }
  if (coords[1] != 0) {
    memcpy(Bloc, Brecv, sizeof(int) * bsize * bsize);
  }

  // test the result (locally), barrier to get the results sequentially
  for (i = 0; i < size; i++) {
    MPI_Barrier(grid_comm);
    if (rank == i) {
      printf("-- [%d] Result --\n", i);
      print_array(Cloc, bsize);
    }
    MPI_Barrier(grid_comm);
  }

  // no gather at the end


  MPI_Finalize();
}
