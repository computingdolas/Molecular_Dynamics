#include "Parser.h"

//Read parameter names, their values and store in in a string to string map
void Parser::readParameters(){
    std::string param_name;
    std::string param_value;

    std::ifstream inputfile;
    inputfile.open(this->filename);

    if(!inputfile.is_open()){
        std::cerr<<"Could not open "<<this->filename<<std::endl;
    }
    for(int i=0;i<this->num_params;i++){
        inputfile>>param_name>>param_value;
        params[param_name] = param_value;
    }
    inputfile.close();
}

//Read input configuration and return the number of particles
void Parser::readInputConfiguration(){
    std::string config_filename;

    real_d mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z;

    config_filename = this->params["part_input_file"];

    std::ifstream input_file;

    input_file.open(config_filename);

    if(!input_file.is_open()){
        std::cerr<<"Could not open file "<<config_filename;
    }
    else{
        input_file>>this->num_particles;
    }
    for(int i=0;i<this->num_particles;i++){
        input_file>>mass>>pos_x>>pos_y>>pos_z>>vel_x>>vel_y>>vel_z;
        (this->mass).push_back(mass);
        (this->pos_x).push_back(pos_x);
        (this->pos_y).push_back(pos_y);
        (this->pos_z).push_back(pos_z);
        (this->vel_x).push_back(vel_x);
        (this->vel_y).push_back(vel_y);
        (this->vel_z).push_back(vel_z);
    }

    input_file.close();
}
