#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "layers.hpp"
#include <vector>
#include "tensor_load.hpp"
#include <algorithm>



class NeuralNet:{
    public:
        std::vector<Layer> layers_class;
        NeuralNet(std::vector<Layer>& layers):{
        layers_class = layers;
        }

        Tensor forward(Tensor& inputs){
            for(const auto& layer : layers_class)
                inputs = layer->forward(inputs)
                std::cout << "yes" << std::endl;
            return inputs;
        }

        Tensor backward(Tensor& grad){
            auto reversed = std::reverse(layers_class.begin(),layers_class.end());
            for(const auto& layer : reversed)
                grad = layer->backward(grad);
            return grad;
        }
};
#endif       



    