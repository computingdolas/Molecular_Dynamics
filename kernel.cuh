#include <cuda_runtime.h>
#include <vector>
#include "Type.h"


//Calculation of the Leonard Jones Potential
__device__ void lenardJonesPotential(const real_d *relativeVector,real_d * forceVector ,const real_d sigma , const real_d epislon) ;



// Calculation of the forces
__global__ void  calcForces(real_d *force,real_d *position,const real_l numParticles ,const real_d sigma, const real_d epislon){

    real_l idx = threadIdx.x + blockIdx.x * blockDim.x ;
    real_l vidx = 0 ;\

    // Relative Vector
    real_d relativeVector[3] = {0.0,0.0,0.0} ;

    //Force Vector
    real_d forceVector[3] = {0.0,0.0,0.0} ;

    for (real_l i = 0 ; i < numParticles ; ++i){

        // Find out the index of particle
        vidx = i * 3 ;

        relativeVector[3] = {0.0,0.0,0.0} ;
        // Find out the realtive vector
        relativeVector[0] = position[idx] - position[vidx] ;
        relativeVector[1] = position[idx+1] - position[vidx+1] ;
        relativeVector[2] = position[idx+2] - position[vidx+2] ;

        // Find put the force between these tow particle
        lenardJonesPotential(relativeVector,forceVector ,sigma,epislon) ;
        force[idx] += forceVector[0] ;
        force[idx+1] += forceVector[1] ;
        force[idx+2] += forceVector[2] ;
    }

}

// Position update
__global__ void updatePosition(const real_d *force,real_d *position,const real_d* velocity, const real_d * mass, const real_l numParticles,const real_d timestep ) {

    real_l idx = threadIdx.x + blockIdx.x * blockDim.x ;
    real_l vidx = idx * 3 ;
    position[vidx] +=  (timestep * velocity[vidx] ) + ( (force[vidx] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
    position[vidx+1] +=  (timestep * velocity[vidx+1] ) + ( (force[vidx+1] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
    position[vidx+2] +=  (timestep * velocity[vidx+2] ) + ( (force[vidx+2] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
}

// Velocity Update
__global__ void updateVelocity(const real_d*forceNew,const real_d*forceOld,real_d * velocity, const real_d* mass, const real_d timestep ){

    real_l idx = threadIdx.x + blockIdx.x * blockDim.x ;
    real_l vidx = idx * 3 ;

    velocity[vidx] += ( (forceNew[vidx] + forceOld[vidx]) * timestep ) / (2.0 * mass[idx] ) ;
    velocity[vidx+1] += ( (forceNew[vidx+1] + forceOld[vidx+1]) * timestep ) / (2.0 * mass[idx] ) ;
    velocity[vidx+2] += ( (forceNew[vidx+2] + forceOld[vidx+1]) * timestep ) / (2.0 * mass[idx] ) ;
}

// Calculation of Leonard Jones Potential
__device__ void lenardJonesPotential(const real_d *relativeVector,real_d * forceVector ,const real_d sigma , const real_d epislon) {

    forceVector[3] = {0.0} ;

    real_d distmod = sqrt( (relativeVector[0]* relativeVector[0]) + (relativeVector[1] * relativeVector[1]) + relativeVector[2] * relativeVector[2] ) ;
    real_d dist = distmod * distmod ;
    real_d sigmaConstant =  sigma / distmod ;
    real_d epislonConstant =  (24.0 * epislon) / dist ;

    forceVector[0] = epislonConstant * pow(sigmaConstant,6.0) * (2*(pow(sigmaConstant,6.0)) - 1 ) * relativeVector[0] ;
    forceVector[1] = epislonConstant * pow(sigmaConstant,6.0) * (2*(pow(sigmaConstant,6.0)) - 1 ) * relativeVector[1] ;
    forceVector[2] = epislonConstant * pow(sigmaConstant,6.0) * (2*(pow(sigmaConstant,6.0)) - 1 ) * relativeVector[2] ;
}


