#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "tensor_load.hpp"
#include "layers.hpp"


class Tanh : public Layer{
    public:

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


        Tensor sigmoid(Tensor& x){
            /*The irrational number e is also known as Euler’s number. 
               It is approximately 2.718281, and is the base of the natural logarithm, ln (this means that, if , then . 
               For real input, exp(x) is always positive.*/
            // Calculate the exponential of all elements in the input array and add 1
            auto denom_sigmoid = 1 + xt::exp(x); 
            return 1 / denom_sigmoid;
        }
};

class Relu : public Layer{
    public:


        Tensor relu(Tensor& x){
            // maximum of x and 0 is relu
            Tensor rectified_tensor = xt::maximum(x,0);
            return rectified_tensor;
        }
};

class Softmax : public Layer{
    public:


        Tensor softmax(Tensor& x){
            /*The softmax function takes as input a vector z of K real numbers, 
            and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers. 
            That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; 
            but after applying softmax, each component will be in the interval 
            (0,1)and the components will add up to 1, so that they can be interpreted as probabilities.*/
            auto exp_values = xt::exp(x);
            auto exp_values_sum = xt::sum(exp_values,1);
            return  exp_values / exp_values_sum;
        }
};

#endif