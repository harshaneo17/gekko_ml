#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor_load.hpp"
#include <map>
#include <typeinfo>

class Layer {
    public:
        
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
        Linear (double input_size,double output_size) : Layer() {
            double input_class_size = input_size;
            double output_class_size = output_size;
             //describe the constructor outside
        Tensor print_v = xt::random::randn<double>({input_class_size,output_class_size});
        Tensor print_g = xt::random::randn<double>({output_class_size});
        Tensor inputs_class;
        }
        
        Tensor forward(Tensor& inputs) override {
            /*outputs = inputs @ w + b*/
            inputs_class = inputs;
            Tensor sum = print_v + print_g;
            std::cout << "this is sum" << sum << std::endl;
            Tensor aids = inputs * sum;
            return aids;
        }

        // Tensor backward(Tensor& grad) override {
        //     /**/
        //     grads["b"] = xt::sum(grad, 1);
        //     auto tr_inputs = xt::transpose(inputs_class);
        //     grads["w"] = xt::operator*(tr_inputs,grad);
        //     auto tr_grad_w = xt::transpose(grads["w"]);
        //     return xt::operator*(grad,tr_grad_w);
        // }

};


#endif

