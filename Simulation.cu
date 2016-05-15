#include <iostream>
#include <list>
#include "cudaDeviceBuffer.h"
#include <cuda_runtime.h>

__global__  void modifier(double *a,double *b,double *c){
        unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
        a[idx] = 1.0;
        b[idx] = 2.0;
        c[idx] = 3.0;

}
#define num_part 256

int main(int argc, const char * argv[]) {
    PhysicalQuantity qua = Vector;
    cudaDeviceBuffer<double> vec1(num_part,qua),vec2(num_part,qua),vec3(num_part,qua);

    vec1.allocateOnDevice();vec1.copyToDevice();
    vec2.allocateOnDevice();vec2.copyToDevice();
    vec3.allocateOnDevice();vec3.copyToDevice();

    modifier<<<12,64>>>(vec1.devicePtr,vec2.devicePtr,vec3.devicePtr);

    cudaDeviceSynchronize();

    vec1.copyToHost();
    vec2.copyToHost();
    vec3.copyToHost();

    for(int i=0;i<256*3;i++){
	std::cout<<vec1[i]<<" "<<vec2[i]<<" "<<vec3[i]<<std::endl;
 }
    return 0;
}                           
