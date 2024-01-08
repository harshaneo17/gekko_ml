#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor_load.hpp"

class Layer {
    public:
        Layer{};

        virtual xt::xtensor forward(xt::xtensor& inputs){
            std::cout << "Not Implemented";
        }

        virtual xt::xtensor backward(xt::tensor& grad){
            std::cout << "Not Implemented"
        }
};

class Linear : public Layer {

    /*computes output = inputs @ weights + biases*/


};

#endif