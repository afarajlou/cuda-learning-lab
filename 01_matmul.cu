// Lesson 1: C = A*B, square N x N matrices, row-major.
// Each 16x16 block computes one tile of C with shared-memory reuse.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
constexpr int N=32, TILE=16;
__global__ void matmul(const float* A,const float* B,float* C){
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row=blockIdx.y*TILE+threadIdx.y;
    int col=blockIdx.x*TILE+threadIdx.x;
    float sum=0;
    for(int base=0;base<N;base+=TILE){
        As[threadIdx.y][threadIdx.x]=(row<N && base+threadIdx.x<N)?A[row*N+base+threadIdx.x]:0;
        Bs[threadIdx.y][threadIdx.x]=(col<N && base+threadIdx.y<N)?B[(base+threadIdx.y)*N+col]:0;
        __syncthreads(); // finish tile loads
        for(int k=0;k<TILE;++k) sum+=As[threadIdx.y][k]*Bs[k][threadIdx.x];
        __syncthreads(); // readers finish before next tile overwrite
    }
    if(row<N && col<N) C[row*N+col]=sum;
}
int main(){
    float *A,*B,*C;
    CHECK(cudaMallocManaged(&A,N*N*sizeof(float)));
    CHECK(cudaMallocManaged(&B,N*N*sizeof(float)));
    CHECK(cudaMallocManaged(&C,N*N*sizeof(float)));
    for(int i=0;i<N;++i)for(int j=0;j<N;++j){A[i*N+j]=(i==j)?1.0f:0.0f;B[i*N+j]=float((i+j)%7);}
    dim3 block(TILE,TILE),grid((N+TILE-1)/TILE,(N+TILE-1)/TILE);
    matmul<<<grid,block>>>(A,B,C);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    int errors=0;
    for(int i=0;i<N*N;++i)if(fabsf(C[i]-B[i])>1e-5f)++errors;
    printf("Tiled matmul (I*B=B): %s\n",errors?"FAIL":"PASS");
    CHECK(cudaFree(A)); CHECK(cudaFree(B)); CHECK(cudaFree(C));
    return errors?1:0;
}
