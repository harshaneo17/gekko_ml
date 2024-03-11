#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor_load.hpp"

struct Params{
  Tensor weights;
  Tensor bias;
  Tensor grad_weights;
  Tensor grad_biases;
};

class Layer {
    public:
        virtual Tensor forward(Tensor inputs) {}
        virtual Tensor backward(Tensor grad, Tensor inputs){}
};

class Linear : public Layer {
    public:
        /*computes output = inputs @ weights + biases*/
        Linear (double input_size,double output_size) : input_class_size(input_size),output_class_size(output_size){}
        Tensor weights,bias,grad_weights,grad_bias;
        Params params;

        void initialize(){};
        
        Tensor forward(Tensor inputs) override {};

        Tensor backward(Tensor grad, Tensor inputs) override {};
        private:
            double input_class_size;
            double output_class_size;

};


#endif

