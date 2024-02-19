#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "layers.hpp"
#include <vector>
#include "tensor_load.hpp"
#include <algorithm>



class NeuralNet{
    public:
        std::vector<Linear> layers_class;
        NeuralNet(std::vector<Linear>& layers):layers_class(layers) {}

        Tensor forward(Tensor& inputs){
            for(auto& layer : layers_class)
                inputs = layer.forward(inputs);
            std::cout << inputs << std::endl;
            return inputs;
        }

        Tensor backward(Tensor& grad){
            std::vector<Linear> rev_layers_class = layers_class; 
            std::reverse(rev_layers_class.begin(),rev_layers_class.end());
            for(auto& layer : rev_layers_class)
                grad = layer.backward(grad);
            return grad;
        }

        std::vector<std::tuple<Tensor,Tensor>> params_and_grads() {
            std::vector<std::tuple<Tensor,Tensor>> result;
            for (auto& layer : layers_class) {
                result.push_back(std::make_tuple(layer.params.weights,layer.params.bias));
                result.push_back(std::make_tuple(layer.params.grad_weights,layer.params.grad_biases));
            }
            return result;
        }
};


#endif       



    