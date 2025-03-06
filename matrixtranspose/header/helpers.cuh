#pragma once

#include <utility>
#include <random>
#include <algorithm>

#define getLasterror                                                                                            \
  do                                                                                                            \
  {                                                                                                             \
    auto err = cudaGetLastError();                                                                              \
    if (err != cudaSuccess)                                                                                     \
    {                                                                                                           \
      printf("check previous kernel %s in line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
    }                                                                                                           \
  } while (0)

#define cudaerrchk(call)                                                                         \
  do                                                                                             \
  {                                                                                              \
    auto err_t = call;                                                                           \
    if (err_t != cudaSuccess)                                                                    \
      printf("error : %s on line %d in file %s\n", cudaGetErrorString(err_t), __LINE__, __FILE__); \
                                                                                                 \
  } while (0)

std::pair<dim3, dim3> getGpuLaunchConfig(const size_t N)
{
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  int warpsize = 0;
  int numSms = 0;
  cudaerrchk(cudaDeviceGetAttribute(&warpsize, cudaDevAttrWarpSize, deviceId));
  cudaerrchk(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, deviceId));
  dim3 thperblk(warpsize * 6, 4);
  dim3 numblocks((N + thperblk.x + 1) / thperblk.x, (N + thperblk.y + 1) / thperblk.y);
  return {numblocks, thperblk};
}

template <typename VT>
class host_dev_arr
{
public:
  host_dev_arr(size_t N, bool fillrand = false) : dev(cudadev(N)), host(cudahost(N)), numelem(N)
  {
    if (fillrand)
    {
      printf("filling in random values\n");
      std::random_device rd;
      std::mt19937 gen(rd());                                // Mersenne Twister RNG
      std::uniform_real_distribution<double> dist(0.0, 1.0); // Range 1-100
      std::generate(host, host + N, [&]()
                    { return dist(gen); });
    }
    else
      std::fill(this->host, this->host + N, static_cast<VT>(0));
  }

  void toDevice()
  {
    cudaerrchk(cudaMemcpy(this->dev, this->host, numelem * sizeof(VT), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    getLasterror;
  }

  void toHost()
  {
    cudaerrchk(cudaMemcpy(this->host, this->dev, numelem * sizeof(VT), cudaMemcpyDeviceToHost));
  }

  VT *getHostPtr()
  {
    return this->host;
  }

  VT *getDevPtr()
  {
    return this->dev;
  }

  size_t getcount() const {
    return numelem;
  }

  size_t getsizeinbytes() const {
    return numelem * sizeof(VT);
  }

  ~host_dev_arr()
  {
    cudaerrchk(cudaFree(this->dev));
    cudaerrchk(cudaFreeHost(this->host));
  }

private:
  VT *cudahost(size_t N)
  {
    VT *hs;
    cudaerrchk(cudaMallocHost((void **)&hs, N * sizeof(VT)));
    return hs;
  }

  VT *cudadev(size_t N)
  {
    VT *ds;
    cudaerrchk(cudaMalloc((void **)&ds, N * sizeof(VT)));
    return ds;
  }
  VT *dev;
  VT *host;
  size_t numelem;
};
