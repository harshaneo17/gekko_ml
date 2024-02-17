#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "layers.hpp"
#include <vector>
#include "tensor_load.hpp"
#include <algorithm>



class NeuralNet{
    public:
        std::vector<Linear> layers_class;
        NeuralNet(std::vector<Linear>& layers):{
        layers_class = layers;
        }

        Tensor forward(Tensor& inputs){
            for(auto& layer : layers_class)
                inputs = layer.forward(inputs);
                std::cout << "yes" << std::endl;
            return inputs;
        }

        Tensor backward(Tensor& grad){
            auto reversed = std::reverse(layers_class.begin(),layers_class.end());
            for(auto& layer : reversed)
                grad = layer.backward(grad);
            return grad;
        }

        // std::vector<std::tuple<Tensor*, Tensor*>> params_and_grads(){
        //     std::vector<std::tuple<Tensor*, Tensor*>> result;
        //     for (const auto& layer : layers_class) {
        //         for (size_t i = 0; i < layer.params.size(); ++i) {
        //             Tensor* param = &layer.params[i];
        //             Tensor* grad = &layer.grads[i];
        //             result.push_back(std::make_tuple(param, grad));
        //         }
        //     }
        //     std::cout << "return results" << result << std::endl;
        //     return result;
        //     }
};


#endif       



    