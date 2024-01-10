#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor_load.hpp"
#include <map>

class Layer {
    public:
        Layer{};
        std::map<std::string,xt::xtensor> params;
        std::map<std::string,xt::xtensor> grads;

        virtual xt::xtensor forward(xt::xtensor& inputs){
            std::cout << "Not Implemented";
        }

        virtual xt::xtensor backward(xt::tensor& grad){
            std::cout << "Not Implemented";
        }
};

class Linear : public Layer {
    public:
        
        /*computes output = inputs @ weights + biases*/
        Linear (const input_s,const output_s) : (input_size,output_size) {} //describe the constructor outside

        params["w"] = xt::random::randn(input_size,output_size);
        params["b"] = xt::random::randn(output_size);
        xt::xtensor inputs_class;

    xt::xtensor forward(xt::xtensor& inputs){
        /*outputs = inputs @ w + b*/
        inputs_class = inputs;
        auto sum = params["w"] + params["b"];
        return xt::operator*(inputs,sum);
    }

    xt::xtensor backward(xt::tensor& grad) {
        /**/
        grads["b"] = xt::sum(grad, axis=0);
        auto tr_inputs = xt::transpose(inputs_class);
        grads["w"] = xt::operator*(tr_inputs,grad);
        tr_grad_w = xt::transpose(grads["w"]);
        return xt::operator*(grad,tr_grad_w);
    }

    private:
        double input_size;
        double output_size;

};  

#endif