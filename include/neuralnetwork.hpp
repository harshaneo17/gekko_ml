#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP


#include "tensor_load.hpp"
#include "layers.hpp"


class NeuralNet{
    public:
        std::vector<Layer> layers_class;
        NeuralNet(std::vector<Layer>& layers):layers_class(layers) {}

        Tensor forward(Tensor inputs){
            for(auto layer : layers_class)
                inputs = layer.forward(inputs);
            return inputs;
        }

        Tensor backward(Tensor grad,Tensor inputs){
            std::vector<Layer> rev_layers_class = layers_class; 
            std::reverse(rev_layers_class.begin(),rev_layers_class.end());
            for(auto layer : rev_layers_class)
                grad = layer.backward(grad,inputs);
            return grad;
        }

        std::vector<BatchTuple> params_and_grads() {
            std::vector<BatchTuple> result;
            for (auto layer : layers_class) {
                result.push_back(std::make_tuple(layer.params.weights,layer.params.bias));
                result.push_back(std::make_tuple(layer.params.grad_weights,layer.params.grad_biases));
            }
            return result;
        }
};


#endif       



    