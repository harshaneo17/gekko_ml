#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor_load.hpp"
#include <map>

class Layer {
    public:
        
        std::map<std::string,Tensor> params;
        std::map<std::string,Tensor> grads;

        virtual Tensor forward(Tensor& inputs){
            std::cout << "Not Implemented";
        }

        virtual Tensor backward(Tensor& grad){
            std::cout << "Not Implemented";
        }
};

class Linear : public Layer {
    public:
        
        /*computes output = inputs @ weights + biases*/
        Linear (double input_size,double output_size) : Layer() { //describe the constructor outside
        params["w"] = xt::random::randn<double>(input_size,output_size),
        params["b"] = xt::random::randn<double>(output_size);
        }
        Tensor inputs_class;


    Tensor forward(Tensor& inputs) override {
        /*outputs = inputs @ w + b*/
        inputs_class = inputs;
        auto sum = params["w"] + params["b"];
        return xt::operator*(inputs,sum);
    }

    Tensor backward(Tensor& grad) override {
        /**/
        grads["b"] = xt::sum(grad, 1);
        auto tr_inputs = xt::transpose(inputs_class);
        grads["w"] = xt::operator*(tr_inputs,grad);
        auto tr_grad_w = xt::transpose(grads["w"]);
        return xt::operator*(grad,tr_grad_w);
    }

};

class Tanh : public Layer{
    public:
        Tanh()

        Tensor tanh(Tensor& x){
            return xt::tanh(x);
        }

        Tensor tanh_prime(Tensor& x){
            auto y = tanh(x);
            return 1 - pow(y,2);
        }
};

class Sigmoid : public Layer{
    public:
        Sigmoid()

        Tensor sigmoid(Tensor& x){
            /*The irrational number e is also known as Euler’s number. 
               It is approximately 2.718281, and is the base of the natural logarithm, ln (this means that, if , then . 
               For real input, exp(x) is always positive.*/
            // Calculate the exponential of all elements in the input array and add 1
            auto denom_sigmoid = 1 + xt::exp(x); 
            return 1 / denom_sigmoid;
        }
};

#endif

