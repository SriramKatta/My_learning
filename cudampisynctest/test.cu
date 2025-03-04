#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                               \
  {                                                                                        \
    cudaError_t cuErr = call;                                                              \
    if (cudaSuccess != cuErr)                                                              \
    {                                                                                      \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr)); \
      exit(0);                                                                             \
    }                                                                                      \
  }

enum TAGS
{
  tag1 = 17,
  tag2 = 23
};

void check_vals(int *ptr, size_t N, int checkval)
{

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (size_t i = 0; i < N; i++)
  {
    if (ptr[i] != checkval)
    {
      printf("the arrays are not right with val %d at index %lu in rank %d \n", ptr[i], i, rank);
      MPI_Finalize();
      exit(1);
      return;
    }
  }
  printf("the arrays are perfect in rank %d\n", rank);
}

__global__ void kern(int *ptr, size_t N)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < N;
       i += blockDim.x * gridDim.x)
  {

    float temp = 0.0f;
    for (int j = 0; j < 130000; j++)
    {
      temp = sinf(j) * cosf(j); // Some dummy floating-point operations
    }

    if (temp > 5000)
      printf("sorry");

    ptr[i] += 1.0f;
  }
}

int main(int argc, char **argv)
{

  if (argc != 3)
  {
    printf("useage  : mpirun -n 2 %s <num of ints> <sync: y/n>\n", argv[0]);
    exit(1);
  }

  MPI_Init(&argc, &argv);
  int size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size != 2)
  {
    printf("this program requires only 2 ranks\n");
    MPI_Finalize();
    exit(1);
  }

  unsigned long long size_of_message = atoll(argv[1]);

  int num_devices = 0;
  cudaErrorCheck(cudaGetDeviceCount(&num_devices));
  cudaErrorCheck(cudaSetDevice(rank % num_devices));

  int devid;
  cudaErrorCheck(cudaGetDevice(&devid));
  printf("cuda device id : %d \n", devid);
  int threadsPerBlock = 32 * 16;
  int blocks = 0;
  cudaErrorCheck(cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, devid))

      int *host_buff = NULL;
  cudaErrorCheck(cudaMallocHost(&host_buff, sizeof(int[size_of_message])));

  for (size_t i = 0; i < size_of_message; ++i)
  {
    host_buff[i] = rank * 100;
  }

  int *device_buff;
  int *device_buff2;
  cudaErrorCheck(cudaMalloc(&device_buff, sizeof(int[size_of_message])));
  cudaErrorCheck(cudaMalloc(&device_buff2, sizeof(int[size_of_message])));
  cudaErrorCheck(cudaMemcpy(device_buff, host_buff, sizeof(int[size_of_message]), cudaMemcpyHostToDevice));

  MPI_Status status;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaStream_t stream = 0;
  cudaStreamCreate(&stream);

  cudaEventRecord(start);
  kern<<<blocks, threadsPerBlock, 0, stream>>>(device_buff, size_of_message);
  cudaEventRecord(stop);

  if (argv[2][0] == 'y')
  {
    printf("synchronized before mpi send recv \n");
    cudaStreamSynchronize(stream);
  }
  else
  {
    printf("not synchronized before mpi send recv \n");
  }
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("time taken by kernel %f ms \n", milliseconds);
  if (rank == 0)
  {
    MPI_Send(device_buff, size_of_message, MPI_INT, 1, tag1, MPI_COMM_WORLD);
    MPI_Recv(device_buff2, size_of_message, MPI_INT, 1, tag2, MPI_COMM_WORLD, &status);
  }
  else if (rank == 1)
  {
    MPI_Recv(device_buff2, size_of_message, MPI_INT, 0, tag1, MPI_COMM_WORLD, &status);
    MPI_Send(device_buff, size_of_message, MPI_INT, 0, tag2, MPI_COMM_WORLD);
  }

  double st = MPI_Wtime();
  cudaErrorCheck(cudaMemcpy(host_buff, device_buff2, sizeof(int[size_of_message]), cudaMemcpyDeviceToHost));
  double dur = MPI_Wtime() - st;
  printf("time taken by copy %f ms \n", dur * 1e3);
  if (rank == 0)
  {
    check_vals(host_buff, size_of_message, 101);
  }
  else if (rank == 1)
  {
    check_vals(host_buff, size_of_message, 1);
  }

  MPI_Finalize();
  return 0;
}
