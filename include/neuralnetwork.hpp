#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP


#include "tensor_load.hpp"
#include "layers.hpp"


class NeuralNet{
    public:
        std::vector<Linear> layers_class;
        NeuralNet(std::vector<Linear>& layers):layers_class(layers) {}

        Tensor forward(Tensor inputs){};

        Tensor backward(Tensor grad,Tensor inputs){};

        std::vector<TensorTuple> params_and_grads() {};
};


#endif       



    