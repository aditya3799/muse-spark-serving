#include <cuda_runtime.h>
#include <stdint.h>
__device__ __constant__ int pos[4][2][4]={{{6,7,8,15},{22,23,24,31}},{{3,4,5,12},{19,20,21,28}},{{0,1,2,9},{16,17,18,25}},{{30,26,27,29},{14,10,11,13}}};
__device__ int canonical(int r,bool up){
 if(!up)return r;
 int i=r/256*128+r%256/128*64+r%64,z=i%64;
 return i/64*64+16*((z/2)%4)+4*(z/16)+2*((z/8)%2)+z%2+(r/64%2)*4096;
}
__device__ unsigned code(const unsigned* w,int e,int n,int k,int N,bool up){
 int p=up?n%4096/8*16+n%8+n/4096*8:n;
 int tile=((e*(N/256)+p/256)*16+k/256)*4+p%256/64;
 int ix=k%256/64*512+p%64/16*128+p%8*16+k%64/16*4+k%8/2;
 unsigned v=w[tile*2304+ix];int r=p/8%2+k/8%2*2,h=k%2;
 unsigned mag=(v>>((6+3*r+16*h)%32))&7,sign=((v>>((15+3*r+16*h)%32))&1)<<3;
 unsigned special=((v>>(4+16*h))&1)|(((v>>(16*h))&3)<<1)|((v>>(16*h))&8);
 return r==3?special:mag|sign;
}
__global__ void pack_words(const unsigned* in,unsigned* out,int N,int units){
 int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=units*128)return;
 int u=z/128,w=z%128,g=w/16,q=w/4%4,kk=w%4;
 int tile=u%16,C=u/16%64,G=u/1024%(N/256),e=u/(1024*(N/256));unsigned packed=0;
 #pragma unroll
 for(int m=0;m<4;++m){
  int n=canonical(G*256+tile*16+g+(m%2)*8,N==8192);
  #pragma unroll
  for(int h=0;h<2;++h){
   int k=C*64+16*q+4*kk+2*(m/2)+h;unsigned c=code(in,e,n,k,N,N==8192);
   #pragma unroll
   for(int b=0;b<4;++b)packed|=((c>>b)&1)<<pos[m][h][b];
  }
 }
 out[u*144+w]=packed;
}
__global__ void pack_scales(const unsigned char* in,unsigned char* out,int N,int units){
 int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=units*64)return;
 int u=z/64,v=z%64,g=v/8,q=v/2%4,rr=v%2;
 int tile=u%16,C=u/16%64,G=u/1024%(N/256),e=u/(1024*(N/256));
 int n=canonical(G*256+tile*16+g+rr*8,N==8192),group=C*4+q;
 int p=N==8192?n%4096/8*16+n%8+n/4096*8:n;
 int src=((e*(N/256)+p/256)*16+group/16)*4+p%256/64;
 int ix=group%16/4*256+p%64/16*64+p%8*8+group%4*2+p/8%2;
 out[u*576+512+v]=in[src*9216+8192+ix];
}
extern "C" __declspec(dllexport) int pack_raw(const void* in,void* out,int experts,int N,int block,cudaStream_t stream){
 if(experts<1||(N!=4096&&N!=8192)||(block!=128&&block!=256&&block!=512))return -1;
 int units=experts*(N/256)*64*16;
 pack_words<<<(units*128+block-1)/block,block,0,stream>>>((const unsigned*)in,(unsigned*)out,N,units);
 cudaError_t a=cudaGetLastError();if(a!=cudaSuccess)return int(a);
 pack_scales<<<(units*64+block-1)/block,block,0,stream>>>((const unsigned char*)in,(unsigned char*)out,N,units);
 return int(cudaGetLastError());
}
