#include "neuralnetwork.hpp"

Tensor NeuralNet::forward(Tensor inputs){
    for(auto layer : layers_class)
        inputs = layer.forward(inputs);
    return inputs;
}

Tensor NeuralNet::backward(Tensor grad,Tensor inputs){
    std::vector<Linear> rev_layers_class = layers_class; 
    std::reverse(rev_layers_class.begin(),rev_layers_class.end());
    for(auto layer : rev_layers_class)
        grad = layer.backward(grad,inputs);
    return grad;
}

std::vector<TensorTuple> NeuralNet::params_and_grads() {
    std::vector<TensorTuple> result;
    for (auto layer : layers_class) {
        result.push_back(std::make_tuple(layer.params.weights,layer.params.bias));
        result.push_back(std::make_tuple(layer.params.grad_weights,layer.params.grad_biases));
    }
    return result;
}