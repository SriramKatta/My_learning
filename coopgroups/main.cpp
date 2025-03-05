#include <iostream>
#include <cstdlib>
#include <chrono>

using Time = std::chrono::high_resolution_clock;
using mintime = std::nano;

__global__ 
void kernel(size_t N, double *a)
{
  int start = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i <= N; i += stride)
  {
    a[i] = a[i] + 1.0;
  }
}

void checkdata(size_t N, double *h_a, double checkval)
{
  for (size_t i = 0; i < N; i++)
  {
    if (h_a[i] != checkval)
    {
      std::cout << "FAILED" << h_a[i] << std::endl;
      exit(1);
    }
  }
}

int main(int argc, char const *argv[])
{
  size_t N = 0;
  size_t rep = 0;
  size_t blocks = 0;
  size_t threads = 0;
  if (argc == 5)
  {

    N = std::atoi(argv[1]);
    rep = std::atoi(argv[2]);
    blocks = std::atoi(argv[3]);
    threads = std::atoi(argv[4]);
  }
  else
  {
    std::cout << "the useage is\n"
              << argv[0] << " <Num elements> <num kernel repetitions> <num blocks> <threads per block>" << std::endl;
    N = 1 << 10;
    rep = 10;
    blocks = 256;
    threads = 1024;
  }

  std ::cout << "running with \n"
  << "Num elements : "  << N << "\n"
  << "num kernel repetitions : " << rep << "\n"
  << "num blocks : " << blocks << "\n"
  << "threads per block : " << threads << "\n";

      size_t size = N * sizeof(double);

  double *h_a = nullptr;
  double *d_a = nullptr;

  cudaMallocHost(&h_a, size);
  cudaMalloc(&d_a, size);

  for (size_t i = 0; i < N; i++)
  {
    h_a[i] = 0.0;
  }

  auto startHtoD = Time::now();
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  auto endHtoD = Time::now();

  cudaDeviceSynchronize();

  auto kernelstart = Time::now();
  for (size_t i = 1; i <= rep; i++)
  {
    kernel<<<blocks, threads>>>(N, d_a);
  }
  cudaDeviceSynchronize();
  auto kernelend = Time::now();

  auto startDtoH = Time::now();
  cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
  auto endDtoH = Time::now();

  checkdata(N, h_a, rep);

  double Dtohtime = std::chrono::duration<double, mintime>(endDtoH - startDtoH).count() / mintime::den;
  double kerneltime = std::chrono::duration<double, mintime>(kernelend - kernelstart).count() / mintime::den;
  double HtoDtime = std::chrono::duration<double, mintime>(endHtoD - startHtoD).count() / mintime::den;

  double arraydatasizeinGbytes = static_cast<double>(N * sizeof(double)) / std::giga::num;

  std::cout << kerneltime << " " << (arraydatasizeinGbytes * 2 * rep) / kerneltime << std::endl
            << Dtohtime << " " << arraydatasizeinGbytes / Dtohtime << std::endl
            << HtoDtime << " " << arraydatasizeinGbytes / HtoDtime << std::endl;
  cudaFreeHost(h_a);
  cudaFree(d_a);
  return 0;
}