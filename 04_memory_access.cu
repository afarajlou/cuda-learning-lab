// Lesson 4: adjacent versus strided global-memory access.
// Educational demonstration: compare access patterns, NOT a reliable benchmark.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(call) do { cudaError_t err=(call); if(err!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(err));exit(1);} } while(0)
constexpr int N=1024, STRIDE=8;
__global__ void contiguous(const float* x,float* y,int n){
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  if(tid<n) y[tid]=2.0f*x[tid];
}
__global__ void strided(const float* x,float* y,int n,int stride){
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  if(tid<n) y[tid]=2.0f*x[tid*stride];
}
int main(){
  float *x,*y; CHECK(cudaMallocManaged(&x,N*STRIDE*sizeof(float)));
  CHECK(cudaMallocManaged(&y,N*sizeof(float)));
  for(int i=0;i<N*STRIDE;++i)x[i]=float(i);
  contiguous<<<(N+255)/256,256>>>(x,y,N);
  CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  int wrong=0;
  for(int i=0;i<N;++i) if(y[i]!=2.0f*i) ++wrong;
  strided<<<(N+255)/256,256>>>(x,y,N,STRIDE);
  CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  for(int i=0;i<N;++i)if(y[i]!=2.0f*(i*STRIDE)) ++wrong;
  printf("Memory access correctness: %s\n",wrong?"FAIL":"PASS");
  CHECK(cudaFree(x));CHECK(cudaFree(y));
  return wrong?1:0;
}
