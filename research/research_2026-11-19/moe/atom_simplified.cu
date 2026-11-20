#include <cstdio>
#include "humming_atom.hpp"
int main(){long checks=0,bad=0;for(int n:{128,256,4096,8192})for(int k:{64,128}){
 auto layout=make_humming_layout(n,k);for(int r=0;r<n;r++)for(int c=0;c<k;c++){
 int lane=r%8*4+c%8/2;int p=4*(c%2)+2*((c/8)%2)+(r/8)%2;
 int expected=8*((c/16)*(n*2)+(r/64)*128+lane*4+(r%64)/16)+p;
 ++checks;if(layout(r,c,0)!=expected)++bad;
 }}std::printf("CHECKS %ld BAD %ld\n",checks,bad);return bad?1:0;}
