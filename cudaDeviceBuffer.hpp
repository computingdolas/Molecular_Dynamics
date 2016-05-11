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

template <typename type>
class cudaDeviceBuffer {
    
private:
    
    PhysicalQuantity phyVar ;
    type *devicePtr ; // Pointer on the Device ..///
    std::vector<type> data ; // Data on the host ...//
    real_l actualSize ; // Actual size to be allocated ...//
    
public:
    
    // Constructing the Buffer
    explicit cudaDeviceBuffer(real_l numParticles_,const PhysicalQuantity phyVar_) ;
    ~cudaDeviceBuffer() ;
    
    // Accessing the data
    const type& operator[](real_l index_) const ;
    type& operator[](real_l index_) ;
    
    // Memory Operations
    void copyToHost(type *toHost_) ;
    void copyFromHost(type *fromHost_) ;
    
    // Allocate and Deallocate Memory on device
    void allocateOnDevice() ;
    void freeMemoryOnDevice() ;
    
    //Calculate Memory
    real_l const bytesToAllocate() ;
    
    //Reseting the data array
    void reset() ;

};

#endif /* cudaDeviceBuffer_hpp */
