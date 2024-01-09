#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor_load.hpp"

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
        params["w"] = xt::random::randn(input_size,output_size);
        /*computes output = inputs @ weights + biases*/
        Linear {}; //describe the constructor outside


};

#endif