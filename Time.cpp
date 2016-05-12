//
//  Time.cpp
//  Assignment_1
//
//  Created by Sagar Dolas on 02/05/16.
//  Copyright © 2016 Sagar Dolas. All rights reserved.
//

#include "Time.hpp"

using namespace std::chrono;

SIWIR2::Timer::Timer():start_(Clock_t::now()) {

}

void SIWIR2::Timer::reset(){
    start_ = Clock_t::now() ;
}

const double SIWIR2::Timer::elapsed() const {
    
    return duration<double,std::milli>(Clock_t::now() -start_).count() ;
}

