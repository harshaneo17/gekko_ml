#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor_load.hpp"
#include <map>

class Layer {
    public:
        
        std::map<std::string,xt::xtensor<double,2>> params;
        std::map<std::string,xt::xtensor<double,2>> grads;

        virtual xt::xtensor<double,2> forward(xt::xtensor<double,2>& inputs){
            std::cout << "Not Implemented";
        }

        virtual xt::xtensor<double,2> backward(xt::tensor<double,2>& grad){
            std::cout << "Not Implemented";
        }
};

class Linear : public Layer {
    public:
        
        /*computes output = inputs @ weights + biases*/
        Linear (const input_s,const output_s) : (input_size,output_size) {} //describe the constructor outside

        params["w"] = xt::random::randn(input_size,output_size);
        params["b"] = xt::random::randn(output_size);
        xt::xtensor<double,2> inputs_class;

    xt::xtensor<double,2> forward(xt::xtensor<double,2>& inputs) override {
        /*outputs = inputs @ w + b*/
        inputs_class = inputs;
        auto sum = params["w"] + params["b"];
        return xt::operator*(inputs,sum);
    }

    xt::xtensor<double,2> backward(xt::xtensor<double,2>& grad) override {
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