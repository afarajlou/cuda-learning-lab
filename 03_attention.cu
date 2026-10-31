// Lesson 3: attention(q,K,V) = softmax(q*K^T/sqrt(D))*V.
// A single query, one attention head. Serial softmax is intentionally simple.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
constexpr int TOKENS=4,D=4;
__global__ void scores_kernel(const float* q,const float* K,float* scores){
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=TOKENS)return;
    float dot=0;
    for(int d=0;d<D;++d)dot+=q[d]*K[t*D+d];
    scores[t]=dot/2.0f; // sqrt(D) = 2
}
__global__ void softmax_kernel(float* scores){
    if(blockIdx.x || threadIdx.x)return; // learning example, not a parallel reduction
    float maximum=scores[0];
    for(int t=1;t<TOKENS;++t)maximum=fmaxf(maximum,scores[t]);
    float sum=0;for(int t=0;t<TOKENS;++t){scores[t]=expf(scores[t]-maximum);sum+=scores[t];}
    for(int t=0;t<TOKENS;++t)scores[t]/=sum;
}
__global__ void weighted_values(const float* weights,const float* V,float* out){
    int d=blockIdx.x*blockDim.x+threadIdx.x;
    if(d>=D)return;
    float s=0;for(int t=0;t<TOKENS;++t)s+=weights[t]*V[t*D+d];
    out[d]=s;
}
int main(){
    float *q,*K,*V,*w,*out;
    CHECK(cudaMallocManaged(&q,D*sizeof(float)));
    CHECK(cudaMallocManaged(&K,TOKENS*D*sizeof(float)));
    CHECK(cudaMallocManaged(&V,TOKENS*D*sizeof(float)));
    CHECK(cudaMallocManaged(&w,TOKENS*sizeof(float)));
    CHECK(cudaMallocManaged(&out,D*sizeof(float)));
    for(int d=0;d<D;++d)q[d]=0;
    for(int i=0;i<TOKENS*D;++i){K[i]=float(i%3);V[i]=float(i/D+1);}
    scores_kernel<<<1,32>>>(q,K,w); CHECK(cudaGetLastError());
    softmax_kernel<<<1,1>>>(w); CHECK(cudaGetLastError());
    weighted_values<<<1,32>>>(w,V,out);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    // Zero query yields uniform weights; V token rows = 1,2,3,4 => mean 2.5.
    int errors=0;for(int d=0;d<D;++d)if(fabsf(out[d]-2.5f)>1e-5f)++errors;
    printf("Single-query attention first element: %.3f | %s\n",out[0],errors?"FAIL":"PASS");
    CHECK(cudaFree(q));CHECK(cudaFree(K));CHECK(cudaFree(V));CHECK(cudaFree(w));CHECK(cudaFree(out));
    return errors?1:0;
}
