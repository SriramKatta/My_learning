#pragma once

#include <stdio.h>

// as transpose is nothing but an copy operation the performance should be close to this kernel
template <typename VT>
__global__ void matrix_copy(size_t N, VT *__restrict_arr M1, VT *__restrict_arr M2)
{
  const auto row = threadIdx.y + blockDim.y * blockIdx.y;
  const auto col = threadIdx.x + blockDim.x * blockIdx.x;
  if (row >= N || col >= N)
    return;
  const auto linindex = row * N + col;
  M2[linindex] = M1[linindex];
}

// as transpose is nothing but an copy operation the performance should be close to this kernel
template <typename VT>
__global__ void matrix_fillinindex(size_t N, VT *__restrict_arr M1)
{
  const auto row = threadIdx.y + blockDim.y * blockIdx.y;
  const auto col = threadIdx.x + blockDim.x * blockIdx.x;
  if (row >= N || col >= N)
    return;
  const auto linindex1 = row * N + col;
  const auto linindex2 = col * N + row;
  M1[linindex1] = linindex2;
}

template <typename VT>
void trasnspose_cpu(size_t N, VT *__restrict_arr M1, VT *__restrict_arr M2)
{
  for (int row = 0; row < N; row++)
  {
    for (int col = 0; col < N; col++)
    {
      const auto linindex1 = row * N + col;
      const auto linindex2 = col * N + row;
      M2[linindex1] = M1[linindex2];
    }
  }
}

template <typename VT>
__global__ void trasnspose_V0(size_t N, VT *__restrict_arr M1, VT *__restrict_arr M2)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= N || row >= N)
  {
    return;
  }
  M2[col * N + row] = M1[row * N + col];
}

void check_matrix_transpose(size_t N, double *__restrict_arr M1, double *__restrict_arr M2)
{
  for (int row = 0; row < N; row++)
  {
    for (int col = 0; col < N; col++)
    {
      int linindex1 = row * N + col;
      int linindex2 = col * N + row;
      if (std::fabs(M1[linindex1] - M2[linindex2]) > 1e-16)
      {
        printf("matrix are not properly transposed with error at M[%d,%d] of %1.6lf\n", row, col, std::fabs(M1[linindex1] - M2[linindex1]));
        exit(EXIT_FAILURE);
      }
    }
  }
  printf("matrix are properly transposed \n");
}

template <typename VT>
__global__ void printMatrixGpu(size_t N, VT *__restrict_arr M)
{
  const auto row = threadIdx.y + blockDim.y * blockIdx.y;
  const auto col = threadIdx.x + blockDim.x * blockIdx.x;
  if (row > N || col > N)
    return;
  printf("row : %d, col : %d, val : %1.4lf ", row, col, M[row * N + col]);
}

template <typename VT>
void printMatrix(size_t N, VT *M)
{
  printf("\n");
  for (int row = 0; row < N; ++row)
  {
    for (int col = 0; col < N; ++col)
    {
      printf(" %1.4lf ", M[row * N + col]);
    }

    printf("\n");
  }
}