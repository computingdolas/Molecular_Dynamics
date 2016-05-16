#include <iostream>
#include <list>
#include "cudaDeviceBuffer.h"
#include <cuda_runtime.h>
#include "Parser.h"
#include "PhysicalVariable.h"
#include "Type.h"

#define num_part 2

int main(int argc, const char * argv[]) {

    // Creating the device Buffers
    cudaDeviceBuffer<real_d> mass(num_part,PhysicalQuantity::Scalar) ;
    cudaDeviceBuffer<real_d> position(num_part,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> velocity(num_part,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> force(num_part, PhysicalQuantity::Vector) ;

    // Reading from file
    Parser p(10,"stable.par") ;
    p.readParameters();
    p.fillBuffers(mass,velocity,position);

    // Allocating memory on Device
    mass.allocateOnDevice();
    position.allocateOnDevice();
    velocity.allocateOnDevice();
    force.allocateOnDevice();

    //Copy to Device
    mass.copyToDevice();
    position.copyToDevice();
    velocity.copyToDevice();
    force.copyToDevice();



    /*
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
    */
    return 0;
}                           
