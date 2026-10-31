"""CPU-only checks of the three mathematical examples; this does not run CUDA."""
from math import exp, sqrt
def matmul(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
def dense(w,b,x,relu=False):
    result=[sum(row[j]*x[j] for j in range(len(x)))+bias for row,bias in zip(w,b)]
    return [max(0,v) for v in result] if relu else result
def attention(q,k,v):
    scores=[sum(a*b for a,b in zip(q,row))/sqrt(len(q)) for row in k]
    m=max(scores); e=[exp(s-m) for s in scores]; z=sum(e); weights=[x/z for x in e]
    return [sum(weights[t]*v[t][d] for t in range(len(v))) for d in range(len(q))]
def run():
    a=[[float(i==j) for j in range(4)] for i in range(4)]
    b=[[float((i+j)%7) for j in range(4)] for i in range(4)]
    assert matmul(a,b)==b
    x=[1.,2.,3.,4.]
    h=dense([[0.1]*4 for _ in range(3)],[0.]*3,x,True)
    y=dense([[0.5]*3 for _ in range(2)],[0.]*2,h)
    assert all(abs(t-1.5)<1e-6 for t in y)
    q=[0.]*4;k=[[float(i%3) for i in range(4)] for _ in range(4)]
    v=[[float(t+1)]*4 for t in range(4)]
    out=attention(q,k,v)
    assert all(abs(t-2.5)<1e-6 for t in out)
    print("CPU reference: all 3 examples passed (not a CUDA run)")
if __name__=="__main__":run()
