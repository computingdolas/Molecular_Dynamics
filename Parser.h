#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <fstream>
#include "Type.h"
class Parser{
public:
    int num_params;
    int num_particles;
    std::string filename;
    std::vector<real_d> mass;
    std::vector<real_d> pos_x;
    std::vector<real_d> pos_y;
    std::vector<real_d> pos_z;
    std::vector<real_d> vel_x;
    std::vector<real_d> vel_y;
    std::vector<real_d> vel_z;
    std::map<std::string,std::string> params;


    //Parse the parameters
    void readParameters();

    //Read input configuration
    void readInputConfiguration();

    //Constructor
    Parser(int num_params, std::string filename){
        this->num_params = num_params;
        this->filename = filename;
    }
};
#endif // PARSER_H
