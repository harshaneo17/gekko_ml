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
        Linear (double input_size,double output_size) : input_class_size(input_size),output_class_size(output_size){}
        Tensor inputs_class;
        Tensor weights;
        Tensor bias;

        void initialize(){
            weights = xt::random::randn<double>({input_class_size,output_class_size});
            auto bias = xt::random::randn<double>({output_class_size});
            std::cout << weights << std::endl;
            Tensor new_bias = static_cast<Tensor>(bias);
            std::cout << new_bias << std::endl;
        }
        
        Tensor forward(Tensor& inputs) override {
            /*outputs = inputs @ w + b*/
            initialize();
            inputs_class = inputs;
            // Tensor sum = weights + bias;
            std::cout << inputs_class << std::endl;
            // // Tensor aids = inputs * sum;
            // return sum;
        }

        // Tensor backward(Tensor& grad) override {
        //     /**/
        //     grads["b"] = xt::sum(grad, 1);
        //     auto tr_inputs = xt::transpose(inputs_class);
        //     grads["w"] = xt::operator*(tr_inputs,grad);
        //     auto tr_grad_w = xt::transpose(grads["w"]);
        //     return xt::operator*(grad,tr_grad_w);
        // }
        private:
            double input_class_size;
            double output_class_size;

};


#endif

