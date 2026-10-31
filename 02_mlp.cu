// Lesson 2: tiny MLP forward: y = W2 * ReLU(W1*x + b1) + b2.
// One GPU thread per output neuron. No training/backpropagation.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
constexpr int INPUT=4,HIDDEN=3,OUTPUT=2;
__global__ void dense_relu(const float* W,const float* b,const float* x,float* y,int in,int out,bool relu){
    int neuron=blockIdx.x*blockDim.x+threadIdx.x;
    if(neuron>=out)return;
    float s=b[neuron];
    for(int j=0;j<in;++j)s+=W[neuron*in+j]*x[j];
    y[neuron]=relu?fmaxf(s,0.0f):s;
}
int main(){
    float *x,*w1,*b1,*h,*w2,*b2,*y;
    CHECK(cudaMallocManaged(&x,INPUT*sizeof(float)));
    CHECK(cudaMallocManaged(&w1,HIDDEN*INPUT*sizeof(float)));
    CHECK(cudaMallocManaged(&b1,HIDDEN*sizeof(float)));
    CHECK(cudaMallocManaged(&h,HIDDEN*sizeof(float)));
    CHECK(cudaMallocManaged(&w2,OUTPUT*HIDDEN*sizeof(float)));
    CHECK(cudaMallocManaged(&b2,OUTPUT*sizeof(float)));
    CHECK(cudaMallocManaged(&y,OUTPUT*sizeof(float)));
    for(int i=0;i<INPUT;++i)x[i]=float(i+1);
    for(int i=0;i<HIDDEN*INPUT;++i)w1[i]=0.1f;
    for(int i=0;i<HIDDEN;++i)b1[i]=0;
    for(int i=0;i<OUTPUT*HIDDEN;++i)w2[i]=0.5f;
    for(int i=0;i<OUTPUT;++i)b2[i]=0;
    dense_relu<<<1,32>>>(w1,b1,x,h,INPUT,HIDDEN,true);
    CHECK(cudaGetLastError());
    dense_relu<<<1,32>>>(w2,b2,h,y,HIDDEN,OUTPUT,false);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    // x sums to 10; each hidden neuron = 1; each output = 1.5.
    int errors=0;for(int i=0;i<OUTPUT;++i)if(fabsf(y[i]-1.5f)>1e-5f)++errors;
    printf("MLP inference output: %.3f %.3f | %s\n",y[0],y[1],errors?"FAIL":"PASS");
    CHECK(cudaFree(x));CHECK(cudaFree(w1));CHECK(cudaFree(b1));CHECK(cudaFree(h));
    CHECK(cudaFree(w2));CHECK(cudaFree(b2));CHECK(cudaFree(y));
    return errors?1:0;
}
