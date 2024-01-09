#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor_load.hpp"
#include <map>

class Layer {
    public:
        Layer{};
        std::map<std::string,xt::xtensor> params;

        virtual xt::xtensor forward(xt::xtensor& inputs){
            std::cout << "Not Implemented";
        }

        virtual xt::xtensor backward(xt::tensor& grad){
            std::cout << "Not Implemented"
        }
};

class Linear : public Layer {
    public:
        
        /*computes output = inputs @ weights + biases*/
        Linear {}; //describe the constructor outside

        params["w"] = xt::random::randn(input_size,output_size);


};

#endif