#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor_load.hpp"
#include <map>

class Layer {
    public:
        
        std::map<std::string,Tensor> params;
        std::map<std::string,Tensor> grads;
        
        virtual ~Layer(){}
        virtual Tensor forward(Tensor& inputs){
            std::cout << "Not Implemented";
        }

        virtual Tensor backward(Tensor& grad){
            std::cout << "Not Implemented";
        }
};

class Linear : public Layer {
    public:

        Tensor inputs_class;
        /*computes output = inputs @ weights + biases*/
        Linear (double input_size,double output_size) : Layer() { //describe the constructor outside
        params["w"] = xt::random::randn<double>({input_size,output_size});
        params["b"] = xt::random::randn<double>({output_size});
        }
        
        Tensor forward(Tensor& inputs) override {
            /*outputs = inputs @ w + b*/
            inputs_class = inputs;
            auto sum = params["w"] + params["b"];
            auto aids = inputs * sum;
            return aids;
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


#endif

