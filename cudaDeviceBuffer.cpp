//
//  cudaDeviceBuffer.cpp
//  Particle_Simulation
//
//  Created by Sagar Dolas on 11/05/16.
//  Copyright © 2016 Sagar Dolas. All rights reserved.
//

#include "cudaDeviceBuffer.hpp"

template<typename type>
cudaDeviceBuffer<type>::cudaDeviceBuffer(real_l _numParticles,const PhysicalQuantity _phyVar ) : phyVar(_phyVar) {
    
    if (phyVar == PhysicalQuantity::Scalar) {
        actualSize = _numParticles ;
    }
    else
        actualSize = _numParticles * 3 ;
    
    // Allocating the data
    data.resize(actualSize,0.0) ; // May be problem here
    
}

template <typename type>
cudaDeviceBuffer<type>::~cudaDeviceBuffer<type>(){

    // We dont have to anything, freeing the , or may be calling free_memory_on_Device() ;
}

template<typename type>
const type& cudaDeviceBuffer<type>::operator[](real_l index_) const{
    
    return data[index_] ;
    
}

template<typename type>
type& cudaDeviceBuffer<type>::operator[](real_l index_){
    
    return data[index_] ;

}

template<typename type>
void cudaDeviceBuffer<type>::copyFromHost(type *fromHost_){
    
    // copy to device , can be used cudaMemCpy() ;
    

}

template<typename type >
void cudaDeviceBuffer<type>::copyToHost(type *toHost_){
    
    // copy to host , cudaMemCpy()

}

template<typename type>
void cudaDeviceBuffer<type>::allocateOnDevice(){

    // cudaMalloc
    
}

template<typename type>
void cudaDeviceBuffer<type>::freeMemoryOnDevice(){
    
    //cudafree

}

template<typename type>
real_l const cudaDeviceBuffer<type>::bytesToAllocate(){
    
    //calculate number of bytes
    
}

template<typename type>
void cudaDeviceBuffer<type>::reset(){
    
    //

}


