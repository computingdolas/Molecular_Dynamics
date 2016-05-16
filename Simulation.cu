#include <iostream>
#include <list>
#include "cudaDeviceBuffer.h"
#include <cuda_runtime.h>
#include "Parser.h"
#include "PhysicalVariable.h"
#include "Type.h"
#include "kernel.cuh"
#include <string>
#include "VTKWriter.h"

#define num_part 2

int main(int argc, const char * argv[]) {

    // Creating the device Buffers
    cudaDeviceBuffer<real_d> mass(num_part,PhysicalQuantity::Scalar) ;
    cudaDeviceBuffer<real_d> position(num_part,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> velocity(num_part,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> forceold(num_part,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> forcenew(num_part,PhysicalQuantity::Vector) ;

    // Reading from file
    Parser p(10,"stable.par") ;
    p.readParameters();
    p.fillBuffers(mass,velocity,position);

    std::cout<<forcenew[0]<<" "<<forcenew[1]<<" "<<forcenew[2]<<" "<<forcenew[3]<<" "<<forcenew[4]<<" "<<forcenew[5]<<std::endl;

    std::cout<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<velocity[0]<<" "<<velocity[1]<<" "<<velocity[2]<<std::endl ;
    std::cout<<position[3]<<" "<<position[4]<<" "<<position[5]<<" "<<velocity[3]<<" "<<velocity[4]<<" "<<velocity[5]<<std::endl ;

    // Allocating memory on Device
    mass.allocateOnDevice();
    position.allocateOnDevice();
    velocity.allocateOnDevice();
    forceold.allocateOnDevice();
    forcenew.allocateOnDevice();

    //Copy to Device
    mass.copyToDevice();
    position.copyToDevice();
    velocity.copyToDevice();
    forceold.copyToDevice();
    forcenew.copyToDevice();

    // Parameters from the file
    real_d time_end = std::stod(p.params["time_end"]) ;
    real_d timestep_length = std::stod(p.params["timestep_length"]) ;
    real_d epsilon = std::stod(p.params["epsilon"]) ;
    real_d sigma = std::stod(p.params["sigma"]) ;


    std::cout<<timestep_length<<std::endl ;
    std::cout<<sigma<<std::endl ;

    VTKWriter writer("blocks_");

    // Algorithm to follow
    {
        // calculate Initial forces
        calcForces<<<2,1>>>(forcenew.devicePtr,position.devicePtr,num_part,sigma,epsilon) ;
        for(real_l i = 0 ; i < 10; ++i) {

            // Update the Position
            updatePosition<<<1,2>>>(forcenew.devicePtr,position.devicePtr,velocity.devicePtr,mass.devicePtr,timestep_length);

            // Copy the forces
            copyForces<<<1,2>>>(forceold.devicePtr,forcenew.devicePtr);

            // Calculate New forces
            calcForces<<<1,2>>>(forcenew.devicePtr,position.devicePtr,num_part,sigma,epsilon);

            // Update the velocity
            updateVelocity<<<1,2>>>(forcenew.devicePtr,forceold.devicePtr,velocity.devicePtr,mass.devicePtr,timestep_length);

            // copy to host back
            forcenew.copyToHost();
            position.copyToHost();
            velocity.copyToHost();

            writer.writeVTKOutput(mass,position,velocity,num_part);

        }

    }


    // copy to host back
    forcenew.copyToHost();
    position.copyToHost();
    velocity.copyToHost();
    // release memory from device


    //Visualizing the data
    std::cout<<forcenew[0]<<" "<<forcenew[1]<<" "<<forcenew[2]<<" "<<forcenew[3]<<" "<<forcenew[4]<<" "<<forcenew[5]<<std::endl;
    std::cout<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<velocity[0]<<" "<<velocity[1]<<" "<<velocity[2]<<std::endl ;
    std::cout<<position[3]<<" "<<position[4]<<" "<<position[5]<<" "<<velocity[3]<<" "<<velocity[4]<<" "<<velocity[5]<<std::endl ;


    return 0;
}                           
