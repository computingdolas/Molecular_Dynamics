//
//  cudaDeviceBuffer.hpp
//  Particle_Simulation
//
//  Created by Sagar Dolas on 11/05/16.
//  Copyright © 2016 Sagar Dolas. All rights reserved.
//

#ifndef cudaDeviceBuffer_hpp
#define cudaDeviceBuffer_hpp

#include <stdio.h>
#include <vector>
#include "Type.h"
#include "PhysicalVariable.h"
#include <cuda_runtime.h>
#include <iostream>

#define DIM 3  //..dimension of the system we ar working on ..//
template <typename type>
class cudaDeviceBuffer {
    
private:
    
    PhysicalQuantity phyVar ;
    std::vector<type> data ; // Data on the host ...//
    real_l actualSize ; // Actual size to be allocated ...//
    real_l numBytes_ ;
    
public:
    
    
    type *devicePtr ; // Pointer on the Device ..///

    // Constructing the Buffer......//
    explicit cudaDeviceBuffer(real_l numParticles_,const PhysicalQuantity phyVar_) ;
    ~cudaDeviceBuffer() ;
    
    // Accessing the data.....//
    const type& operator[](real_l index_) const ;
    type& operator[](real_l index_) ;
    
    // Memory Operations....//
    void copyToHost() ;
    void copyToDevice() ;
    
    // Allocate and Deallocate Memory on device....//
    void allocateOnDevice() ;
    void freeMemoryOnDevice() ;
    
    //Calculate Memory...//
    void bytesToAllocate() ;
    
    //Reseting the data array....//
    void reset() ;
    
    // Check Error....//
    void checkError(const cudaError_t err) ;
};

#endif /* cudaDeviceBuffer_hpp */
