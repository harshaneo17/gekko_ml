#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP


#include "tensor_load.hpp"
#include "layers.hpp"


class NeuralNet{
    public:
        std::vector<Layer*> layers_class;
        NeuralNet(std::vector<Layer*> layers):layers_class(layers) {}

        Tensor forward(Tensor& inputs){
            for(auto layer : layers_class)
                inputs = layer->forward(inputs);
            return inputs;
        }

        Tensor backward(Tensor& grad,Tensor inputs){
            Tensor res = grad;
            auto iter = layers_class.rbegin();
            auto end = layers_class.rend();
            while(iter != end)
            {
            res = (*iter)->backward(res,inputs);
            ++iter;
            }

            return res;
        }

        std::vector<TensorTuple> params_and_grads() {
            std::vector<TensorTuple> result;
            for (auto layer : layers_class) {
                result.push_back(std::make_tuple(layer->params.weights,layer->params.bias));
                result.push_back(std::make_tuple(layer->params.grad_weights,layer->params.grad_biases));
            }
            return result;
        }
};


#endif       



    